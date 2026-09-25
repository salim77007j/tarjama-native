//! Translation backend v1.3: Turkish -> Arabic via OPUS-MT (Helsinki
//! opus-mt-tr-ar, ONNX int8) with BEAM SEARCH (beam 4, length penalty 0.2),
//! replacing v1.2's greedy decoding - measured noticeably better Arabic.
//!
//! Model layout (models/mt/tr-ar/), two ONNX decoder sessions:
//!   decoder_model.int8.onnx          - no cache, step 1; outputs logits +
//!                                      all 24 KV "present" tensors (self+cross)
//!   decoder_with_past_model.int8.onnx - steps >= 2; takes past_key_values.*,
//!                                      returns logits + 12 self-attention presents.
//! Cross-attention K/V never change after step 1; only self-attention pasts
//! roll (and get re-indexed when beam search reorders hypotheses).

use anyhow::{anyhow, bail, Context as _, Result};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{Session, SessionInputValue, SessionOutputs};
use ort::value::Tensor;
use serde::Deserialize;
use std::path::Path;
use tokenizers::Tokenizer;

/// Mirrors the web app's CTranslate2 beam settings (beam_size=4,
/// length_penalty=0.2), which measurably improve over greedy decoding.
const BEAM_SIZE: usize = 4;
const LENGTH_PENALTY: f32 = 0.2;
/// Marian positional embeddings cap at 512; keep a safety margin.
const MAX_ENC_TOKENS: usize = 448;

#[derive(Deserialize)]
struct Meta {
    decoder_start_token_id: i64,
    eos_token_id: i64,
    #[serde(default)]
    max_length: usize,
    #[serde(default)]
    num_decoder_heads: usize,
    #[serde(default)]
    head_dim: usize,
}

fn load_session(path: &Path, threads: usize) -> Result<Session> {
    Ok(Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(threads)?
        .commit_from_file(path)?)
}

fn find_name<'a>(names: &'a [String], pred: impl Fn(&str) -> bool, what: &str) -> Result<&'a str> {
    names
        .iter()
        .map(|s| s.as_str())
        .find(|n| pred(n))
        .ok_or_else(|| anyhow!("{what} not found in model I/O"))
}

struct Pair {
    enc: Session,
    dec0: Session, // step 1: no cache
    dec1: Session, // steps >= 2: with cache
    tok: Tokenizer,
    enc_in: [String; 2],
    enc_out: String,
    d0_tok: String,
    d0_hidden: String,
    d0_mask: String,
    d1_tok: String,
    d1_mask: String,
    /// dec1 past input name <-> dec0 present output name, self-attention.
    self_cache: Vec<(String, String)>,
    /// dec1 past input name <-> dec0 present output name, cross-attention.
    cross_cache: Vec<(String, String)>,
    logits0: String,
    logits1: String,
    start_id: i64,
    eos_id: i64,
    max_len: usize,
    heads: usize,
    head_dim: usize,
}

impl Pair {
    fn load(dir: &Path, threads: usize) -> Result<Self> {
        let meta: Meta = serde_json::from_reader(
            std::fs::File::open(dir.join("meta.json")).context("meta.json missing")?,
        )?;
        let tok =
            Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("tokenizer.json: {e}"))?;
        let enc = load_session(&dir.join("encoder_model.int8.onnx"), threads)?;
        let dec0 = load_session(&dir.join("decoder_model.int8.onnx"), threads)?;
        let dec1 = load_session(&dir.join("decoder_with_past_model.int8.onnx"), threads)?;

        let enc_in: Vec<String> = enc.inputs.iter().map(|i| i.name.clone()).collect();
        let enc_out = enc
            .outputs
            .first()
            .map(|o| o.name.clone())
            .ok_or_else(|| anyhow!("encoder has no outputs"))?;

        let d0_in: Vec<String> = dec0.inputs.iter().map(|i| i.name.clone()).collect();
        let d1_in: Vec<String> = dec1.inputs.iter().map(|i| i.name.clone()).collect();
        let d0_out: Vec<String> = dec0.outputs.iter().map(|o| o.name.clone()).collect();
        let d1_out: Vec<String> = dec1.outputs.iter().map(|o| o.name.clone()).collect();

        let d0_tok = find_name(
            &d0_in,
            |n| n.ends_with("input_ids") || n.contains("decoder_input"),
            "decoder token input (step 1)",
        )?
        .to_string();
        let d0_hidden = find_name(&d0_in, |n| n.contains("hidden"), "encoder_hidden_states (step 1)")?.to_string();
        let d0_mask = find_name(&d0_in, |n| n.contains("mask"), "attention mask (step 1)")?.to_string();
        let d1_tok = find_name(
            &d1_in,
            |n| n.ends_with("input_ids") || n.contains("decoder_input"),
            "decoder token input (cached)",
        )?
        .to_string();
        let d1_mask = find_name(&d1_in, |n| n.contains("mask"), "attention mask (cached)")?.to_string();
        let logits0 = find_name(&d0_out, |n| n == "logits", "logits (step 1)")?.to_string();
        let logits1 = find_name(&d1_out, |n| n == "logits", "logits (cached)")?.to_string();

        // pair cached-model past inputs with step-1 present outputs by suffix
        let suffix = |n: &str| n.splitn(2, '.').nth(1).unwrap_or_default().to_string();
        let mut self_cache: Vec<(String, String)> = Vec::new();
        let mut cross_cache: Vec<(String, String)> = Vec::new();
        for past in d1_in.iter().filter(|n| n.starts_with("past_key_values")) {
            let present = format!("present.{}", suffix(past));
            if !d0_out.contains(&present) {
                bail!("step-1 model lacks {present} (needed for past {past})");
            }
            if past.contains(".decoder.") {
                self_cache.push((past.clone(), present));
            } else if past.contains(".encoder.") {
                cross_cache.push((past.clone(), present));
            }
        }
        if self_cache.is_empty() || cross_cache.is_empty() {
            bail!("KV cache pairing incomplete (self={} cross={})", self_cache.len(), cross_cache.len());
        }
        if meta.num_decoder_heads == 0 || meta.head_dim == 0 {
            bail!("meta.json missing num_decoder_heads/head_dim");
        }

        Ok(Self {
            enc,
            dec0,
            dec1,
            tok,
            enc_in: [enc_in[0].clone(), enc_in[1].clone()],
            enc_out,
            d0_tok,
            d0_hidden,
            d0_mask,
            d1_tok,
            d1_mask,
            self_cache,
            cross_cache,
            logits0,
            logits1,
            start_id: meta.decoder_start_token_id,
            eos_id: meta.eos_token_id,
            max_len: if meta.max_length == 0 { 256 } else { meta.max_length.min(256) },
            heads: meta.num_decoder_heads,
            head_dim: meta.head_dim,
        })
    }
}

/// Extract one output tensor as (dims, data).
fn take_f32(outs: &SessionOutputs<'_>, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
    let v = outs.get(name).ok_or_else(|| anyhow!("missing output {name}"))?;
    let (shape, data) = v.try_extract_tensor::<f32>()?;
    let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    Ok((dims, data.to_vec()))
}

fn log_softmax_row(row: &[f32]) -> Vec<f32> {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out = vec![0f32; row.len()];
    let mut sum = 0f32;
    for (o, &v) in out.iter_mut().zip(row) {
        let e = (v - max).exp();
        *o = e;
        sum += e;
    }
    let lsum = sum.ln();
    for o in out.iter_mut() {
        *o = o.ln() - lsum; // ln(e^(v-max)) - ln(sum) = log_softmax
    }
    out
}

struct Beam {
    ids: Vec<i64>,
    score: f32,
}

impl Pair {
    /// Beam-search translate for one already-encoded source. Returns target ids.
    fn generate(&mut self, enc_ids: &[i64]) -> Result<Vec<i64>> {
        // ---- encoder ----
        let s_len = enc_ids.len();
        let mask = vec![1i64; s_len];
        let enc_inputs = ort::inputs![
            self.enc_in[0].as_str() => Tensor::from_array(([1usize, s_len], enc_ids.to_vec()))?,
            self.enc_in[1].as_str() => Tensor::from_array(([1usize, s_len], mask.clone()))?,
        ];
        let outs = self.enc.run(enc_inputs)?;
        let (eshape, hidden1) = take_f32(&outs, &self.enc_out)?;
        if eshape.len() != 3 || eshape[1] != s_len {
            bail!("unexpected encoder output shape {eshape:?} (src len {s_len})");
        }
        let h_dim = eshape[2];
        drop(outs);

        // ---- step 1: no-cache decoder (batch 1, full KV output) ----
        let inputs0 = ort::inputs![
            self.d0_tok.as_str() => Tensor::from_array(([1usize, 1usize], vec![self.start_id]))?,
            self.d0_hidden.as_str() => Tensor::from_array(([1usize, s_len, h_dim], hidden1.clone()))?,
            self.d0_mask.as_str() => Tensor::from_array(([1usize, s_len], mask.clone()))?,
        ];
        let outs = self.dec0.run(inputs0)?;
        let (lshape, logits) = take_f32(&outs, &self.logits0)?;
        let vocab = *lshape.last().ok_or_else(|| anyhow!("empty logits"))?;
        let mut self_data: Vec<Vec<f32>> = Vec::new();
        let mut cross_data: Vec<Vec<f32>> = Vec::new();
        for (past_name, present_name) in self.self_cache.iter().chain(self.cross_cache.iter()) {
            let (dims, data) = take_f32(&outs, present_name)?;
            if dims.len() != 4 || dims[0] != 1 || dims[2] == 0 {
                bail!("step-1 present {past_name} has {dims:?}");
            }
            if past_name.contains(".decoder.") {
                self_data.push(data);
            } else {
                cross_data.push(data);
            }
        }
        drop(outs);

        // seed beams: top BEAM_SIZE non-EOS tokens (min length 1)
        let row = &logits[..vocab];
        let lp = log_softmax_row(row);
        let mut first: Vec<(f32, i64)> =
            lp.iter().enumerate().map(|(t, &p)| (p, t as i64)).collect();
        first.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut beams: Vec<Beam> = first
            .into_iter()
            .filter(|(_, tok)| *tok != self.eos_id)
            .take(BEAM_SIZE)
            .map(|(score, tok)| Beam { ids: vec![tok], score })
            .collect();
        if beams.is_empty() {
            bail!("translation produced no tokens");
        }

        let mut finished: Vec<(f32, usize, Vec<i64>)> = Vec::new(); // (raw score, len, ids)
        let mut kv_len: usize = 1; // self-cache sequence length
        let mut self_batch: usize = 1; // self_data batch (1 after step 1)
        let mut steps_since_finish: usize = 0;

        for _step in 1..self.max_len {
            let b = beams.len();
            // ---- build batch feeds for the cached decoder ----
            let last: Vec<i64> = beams.iter().map(|bm| *bm.ids.last().unwrap()).collect();
            let mut mask_b = Vec::with_capacity(b * s_len);
            for _ in 0..b {
                mask_b.extend_from_slice(&mask);
            }
            let mut feeds: Vec<(String, SessionInputValue<'static>)> = Vec::new();
            feeds.push((self.d1_tok.clone(), SessionInputValue::from(
                Tensor::from_array(([b, 1usize], last))?,
            )));
            feeds.push((self.d1_mask.clone(), SessionInputValue::from(
                Tensor::from_array(([b, s_len], mask_b))?,
            )));
            // self pasts: after step 1 all beams share one cache row -> replicate;
            // afterwards self_data is already gathered to the current beam order.
            let row_sz = self.heads * kv_len * self.head_dim;
            for (i, (name, _)) in self.self_cache.iter().enumerate() {
                let d: Vec<f32> = if self_batch == 1 && b > 1 {
                    let mut dd = Vec::with_capacity(b * row_sz);
                    for _ in 0..b {
                        dd.extend_from_slice(&self_data[i][..row_sz]);
                    }
                    dd
                } else {
                    self_data[i].clone()
                };
                feeds.push((name.clone(), SessionInputValue::from(
                    Tensor::from_array(([b, self.heads, kv_len, self.head_dim], d))?,
                )));
            }
            // cross pasts: constant, replicate batch 1 -> b
            for ((name, _), data) in self.cross_cache.iter().zip(cross_data.iter()) {
                let mut d = Vec::with_capacity(data.len() * b);
                for _ in 0..b {
                    d.extend_from_slice(data);
                }
                let cross_len = data.len() / (self.heads * self.head_dim);
                feeds.push((name.clone(), SessionInputValue::from(
                    Tensor::from_array(([b, self.heads, cross_len, self.head_dim], d))?,
                )));
            }

            let outs = self.dec1.run(feeds)?;
            let (lshape, logits) = take_f32(&outs, &self.logits1)?;
            let vocab = *lshape.last().unwrap();
            // new self presents (batch b) + their sequence length
            let mut new_self: Vec<(Vec<f32>, usize)> = Vec::new();
            for (_, present_name) in &self.self_cache {
                let (dims, data) = take_f32(&outs, present_name)?;
                if dims.len() != 4 || dims[0] != b || dims[2] == 0 {
                    bail!("cached present {present_name} has {dims:?} (batch {b})");
                }
                new_self.push((data, dims[2]));
            }
            drop(outs);

            // ---- candidates: rank by GNMT length-penalized score ----
            // (CTranslate2 semantics: ranking actives by score/((5+len)/6)^alpha
            // lets longer sentences compete with short ones - raw-sum ranking
            // makes any early-EOS short output win, truncating subtitles)
            let plen = |l: usize| ((5.0 + l.max(1) as f32) / 6.0).powf(LENGTH_PENALTY);
            // (penalized, raw, parent, token)
            let mut cands: Vec<(f32, f32, usize, i64)> = Vec::with_capacity(b * BEAM_SIZE);
            for bi in 0..b {
                let row = &logits[bi * vocab..(bi + 1) * vocab];
                let lp = log_softmax_row(row);
                let mut top: Vec<(f32, i64)> =
                    lp.iter().enumerate().map(|(t, &p)| (p, t as i64)).collect();
                top.sort_unstable_by(|a, b2| {
                    b2.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut taken = 0;
                for (p, tok) in top {
                    if tok == self.eos_id && beams[bi].ids.is_empty() {
                        continue; // min length 1
                    }
                    let raw = beams[bi].score + p;
                    let cand_len = beams[bi].ids.len() + 1;
                    cands.push((raw / plen(cand_len), raw, bi, tok));
                    taken += 1;
                    if taken >= BEAM_SIZE {
                        break;
                    }
                }
            }
            cands.sort_unstable_by(|a, b2| {
                b2.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            });

            // ---- prune (finished = EOS-completed hypotheses, raw score) ----
            let fin_before = finished.len();
            let mut next: Vec<Beam> = Vec::with_capacity(BEAM_SIZE);
            let mut new_parents: Vec<usize> = Vec::with_capacity(BEAM_SIZE);
            for (_pen, raw, bi, tok) in cands.into_iter() {
                if tok == self.eos_id {
                    let l = beams[bi].ids.len();
                    finished.push((raw, l, beams[bi].ids.clone()));
                    continue;
                }
                if next.len() >= BEAM_SIZE {
                    break;
                }
                let mut ids = beams[bi].ids.clone();
                ids.push(tok);
                next.push(Beam { ids, score: raw });
                new_parents.push(bi);
            }
            if next.is_empty() {
                beams = next;
                break;
            }
            // patience: once we hold a full set of finished hypotheses, stop
            // after a quiet window with no new finisher (generous: longer,
            // better sentences keep finishing well past the first short ones)
            if finished.len() > fin_before {
                steps_since_finish = 0;
            } else {
                steps_since_finish += 1;
                if steps_since_finish >= 2 * BEAM_SIZE && finished.len() >= BEAM_SIZE {
                    beams = next;
                    break;
                }
            }
            beams = next;

            // ---- roll the self cache, reindexed to the new beam order ----
            let kv_new = new_self.first().map(|(_, seq)| *seq).unwrap_or(kv_len + 1);
            let r = self.heads * kv_new * self.head_dim;
            let mut rolled: Vec<Vec<f32>> = Vec::with_capacity(new_self.len());
            for (data, _) in new_self.iter() {
                let mut out = Vec::with_capacity(new_parents.len() * r);
                for &p in &new_parents {
                    out.extend_from_slice(&data[p * r..(p + 1) * r]);
                }
                rolled.push(out);
            }
            self_data = rolled;
            kv_len = kv_new;
            self_batch = new_parents.len();

            if beams.is_empty() {
                break;
            }
        }

        // ---- pick the best EOS-completed hypothesis. Comparing finished
        // (EOS natural stop) against truncated active beams produces cut-off
        // subtitles, so active beams are used ONLY if nothing finished. ----
        let penal = |score: f32, len: usize| -> f32 {
            score / ((5.0 + len.max(1) as f32) / 6.0).powf(LENGTH_PENALTY)
        };
        let best_finished = finished
            .iter()
            .map(|(s, l, ids)| (penal(*s, *l), ids))
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let best_active = if finished.is_empty() {
            beams
                .iter()
                .map(|bm| (penal(bm.score, bm.ids.len()), &bm.ids))
                .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        } else {
            None
        };
        let best = match (best_finished, best_active) {
            (Some(f), Some(a)) => if f.0 >= a.0 { f.1.clone() } else { a.1.clone() },
            (Some(f), None) => f.1.clone(),
            (None, Some(a)) => a.1.clone(),
            (None, None) => bail!("beam search produced no hypothesis"),
        };
        Ok(best)
    }
}

/// Turkish -> Arabic translator (OPUS-MT int8, beam search).
pub struct Translator {
    pair: Pair,
}

impl Translator {
    /// Load from <models_dir>/mt/tr-ar.
    pub fn load(models_dir: &Path) -> Result<Self> {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2)
            .max(1)
            .min(8);
        let pair = Pair::load(&models_dir.join("mt").join("tr-ar"), threads)
            .with_context(|| format!("loading mt/tr-ar from {}", models_dir.display()))?;
        Ok(Self { pair })
    }

    /// Turkish text -> Arabic text, beam search.
    pub fn translate(&mut self, text: &str) -> Result<String> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(String::new());
        }
        let enc = self
            .pair
            .tok
            .encode(text, true)
            .map_err(|e| anyhow!("encode(tr): {e}"))?;
        let mut ids: Vec<i64> = enc.get_ids().iter().map(|&v| v as i64).collect();
        if ids.is_empty() {
            return Ok(String::new());
        }
        ids.truncate(MAX_ENC_TOKENS);
        let ar_ids = self.pair.generate(&ids)?;
        if ar_ids.is_empty() {
            bail!("translation produced no tokens");
        }
        let ar = self
            .pair
            .tok
            .decode(&ar_ids.iter().map(|&v| v as u32).collect::<Vec<_>>(), true)
            .map_err(|e| anyhow!("decode(ar): {e}"))?;
        Ok(ar.trim().to_string())
    }
}
