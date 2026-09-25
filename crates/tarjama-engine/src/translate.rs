use anyhow::{anyhow, bail, Context as _, Result};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use serde::Deserialize;
use std::path::Path;
use tokenizers::Tokenizer;

#[derive(Deserialize)]
struct Meta {
    enc_inputs: Vec<String>,
    enc_outputs: Vec<String>,
    dec_inputs: Vec<String>,
    dec_outputs: Vec<String>,
    decoder_start_token_id: i64,
    eos_token_id: i64,
    max_length: usize,
}

pub struct Translator {
    enc: Session,
    dec: Session,
    tok: Tokenizer,
    enc_in: [String; 2],
    enc_out: String,
    dec_in_ids: String,
    dec_in_hidden: String,
    dec_in_mask: Option<String>,
    dec_out: String,
    start_id: i64,
    eos_id: i64,
    max_len: usize,
}

impl Translator {
    pub fn load(dir: &Path) -> Result<Self> {
        let meta: Meta = serde_json::from_reader(
            std::fs::File::open(dir.join("meta.json")).context("meta.json missing")?,
        )?;
        let tok = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("tokenizer.json: {e}"))?;

        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2)
            .max(1)
            .min(8);
        let mk = |f: &str| -> Result<Session> {
            Ok(Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(threads)?
                .commit_from_file(dir.join(f))?)
        };
        let enc = mk("encoder_model.int8.onnx")?;
        let dec = mk("decoder_model.int8.onnx")?;

        let pick = |want: &[&str], names: &[String]| -> Result<String> {
            for w in want {
                if let Some(n) = names.iter().find(|n| n == w) {
                    return Ok(n.clone());
                }
            }
            bail!("model I/O {want:?} not found in {names:?}");
        };
        let enc_in0 = pick(&["input_ids"], &meta.enc_inputs)?;
        let enc_in1 = pick(&["attention_mask"], &meta.enc_inputs)?;
        let enc_out = meta
            .enc_outputs
            .first()
            .cloned()
            .ok_or_else(|| anyhow!("encoder has no outputs"))?;

        let dec_in_ids = pick(&["decoder_input_ids"], &meta.dec_inputs)?;
        let dec_in_hidden = pick(&["encoder_hidden_states"], &meta.dec_inputs)?;
        let dec_in_mask = meta
            .dec_inputs
            .iter()
            .find(|n| n.contains("encoder_attention_mask"))
            .cloned();
        let dec_out = pick(&["logits"], &meta.dec_outputs)?;

        Ok(Self {
            enc,
            dec,
            tok,
            enc_in: [enc_in0, enc_in1],
            enc_out,
            dec_in_ids,
            dec_in_hidden,
            dec_in_mask,
            dec_out,
            start_id: meta.decoder_start_token_id,
            eos_id: meta.eos_token_id,
            max_len: meta.max_length.min(256),
        })
    }

    pub fn translate(&mut self, text: &str) -> Result<String> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(String::new());
        }
        let enc = self
            .tok
            .encode(text, true)
            .map_err(|e| anyhow!("encode: {e}"))?;
        let ids: Vec<i64> = enc.get_ids().iter().map(|&v| v as i64).collect();
        if ids.is_empty() {
            return Ok(String::new());
        }
        let len = ids.len();
        let mask = vec![1i64; len];

        let enc_inputs = ort::inputs![
            self.enc_in[0].as_str() => Tensor::from_array(([1usize, len], ids))?,
            self.enc_in[1].as_str() => Tensor::from_array(([1usize, len], mask.clone()))?,
        ];
        let outs = self.enc.run(enc_inputs)?;
        let hidden_v = outs
            .get(self.enc_out.as_str())
            .ok_or_else(|| anyhow!("missing encoder output"))?;
        let (shape, hidden) = hidden_v.try_extract_tensor::<f32>()?;
        if shape.len() != 3 {
            bail!("unexpected encoder output shape {shape:?}");
        }
        let s = shape[1] as usize;
        let h = shape[2] as usize;
        let hidden = hidden.to_vec();

        let mut dec_ids: Vec<i64> = vec![self.start_id];
        for _ in 0..self.max_len {
            let cur = dec_ids.len();
            let mut inputs = ort::inputs![
                self.dec_in_ids.as_str() => Tensor::from_array(([1usize, cur], dec_ids.clone()))?,
                self.dec_in_hidden.as_str() => Tensor::from_array(([1usize, s, h], hidden.clone()))?,
            ];
            if let Some(m) = &self.dec_in_mask {
                inputs = ort::inputs![
                    self.dec_in_ids.as_str() => Tensor::from_array(([1usize, cur], dec_ids.clone()))?,
                    self.dec_in_hidden.as_str() => Tensor::from_array(([1usize, s, h], hidden.clone()))?,
                    m.as_str() => Tensor::from_array(([1usize, len], mask.clone()))?,
                ];
            }
            let outs = self.dec.run(inputs)?;
            let logits_v = outs
                .get(self.dec_out.as_str())
                .ok_or_else(|| anyhow!("missing decoder logits"))?;
            let (lshape, logits) = logits_v.try_extract_tensor::<f32>()?;
            if lshape.len() != 3 {
                bail!("unexpected logits shape {lshape:?}");
            }
            let vocab = lshape[2] as usize;
            let last = &logits[(cur - 1) * vocab..cur * vocab];
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, &v) in last.iter().enumerate() {
                if v > bv {
                    bv = v;
                    best = i;
                }
            }
            let next = best as i64;
            if next == self.eos_id {
                break;
            }
            dec_ids.push(next);
        }
        let out_ids: Vec<u32> = dec_ids[1..].iter().map(|&v| v as u32).collect();
        let ar = self
            .tok
            .decode(&out_ids, true)
            .map_err(|e| anyhow!("decode: {e}"))?;
        Ok(ar.trim().to_string())
    }
}
