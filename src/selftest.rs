use crate::pipeline::{self, Kind, Queue};
use anyhow::{anyhow, bail, Result};
use std::collections::VecDeque;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Instant;

const SELFTEST_WAV: &[u8] = include_bytes!("../assets/selftest_tr.wav");

/// Minimal RIFF WAV parser (16-bit PCM -> f32 mono 16 kHz).
pub fn decode_wav(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        bail!("not a RIFF WAVE file");
    }
    let mut pos = 12usize;
    let mut fmt: Option<(u16, u16, u32, u16)> = None; // format, channels, rate, bits
    let mut data: Option<&[u8]> = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes([bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]]) as usize;
        let body_start = pos + 8;
        let body_end = (body_start + size).min(bytes.len());
        if id == b"fmt " {
            let b = &bytes[body_start..body_end];
            if b.len() >= 16 {
                fmt = Some((
                    u16::from_le_bytes([b[0], b[1]]),
                    u16::from_le_bytes([b[2], b[3]]),
                    u32::from_le_bytes([b[4], b[5], b[6], b[7]]),
                    u16::from_le_bytes([b[14], b[15]]),
                ));
            }
        } else if id == b"data" {
            data = Some(&bytes[body_start..body_end]);
        }
        pos = body_start + size + (size % 2);
    }
    let (fmt, channels, rate, bits) = fmt.ok_or_else(|| anyhow!("no fmt chunk"))?;
    let raw = data.ok_or_else(|| anyhow!("no data chunk"))?;
    if fmt != 1 || bits != 16 {
        bail!("only 16-bit PCM supported (fmt={fmt}, bits={bits})");
    }
    let mut samples: Vec<f32> = raw
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
        .collect();
    if channels > 1 {
        samples = samples.chunks(channels as usize).map(|c| c.iter().sum::<f32>() / c.len() as f32).collect();
    }
    if rate != 16000 {
        // naive linear resample
        let step = rate as f64 / 16000.0;
        let n = ((samples.len() as f64) / step) as usize;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let p = i as f64 * step;
            let i0 = p as usize;
            let i1 = (i0 + 1).min(samples.len() - 1);
            let f = p - i0 as f64;
            out.push(samples[i0] * (1.0 - f as f32) + samples[i1] * f as f32);
        }
        samples = out;
    }
    Ok(samples)
}

pub fn run() -> Result<()> {
    println!("Tarjama Studio Native - selftest");
    println!("================================");
    let t_all = Instant::now();

    let mdir = pipeline::models_dir()?;
    println!("models dir: {}", mdir.display());

    for k in Kind::all() {
        let p = mdir.join(k.file());
        let meta = std::fs::metadata(&p).map_err(|_| anyhow!("MISSING model file: {}", p.display()))?;
        println!("  ok {:?} {} ({:.1} MB)", k, p.display(), meta.len() as f64 / 1e6);
    }
    let mt = mdir.join("opus-mt");
    for f in [
        "encoder_model.int8.onnx",
        "decoder_model.int8.onnx",
        "tokenizer.json",
        "meta.json",
    ] {
        let p = mt.join(f);
        let meta = std::fs::metadata(&p).map_err(|_| anyhow!("MISSING translation file: {}", p.display()))?;
        println!("  ok opus-mt/{f} ({:.1} MB)", meta.len() as f64 / 1e6);
    }

    // 1) decode embedded Turkish sample
    let samples = decode_wav(SELFTEST_WAV)?;
    println!("sample: {} samples ({:.1}s)", samples.len(), samples.len() as f64 / 16000.0);
    if samples.len() < 8000 {
        bail!("selftest wav too small");
    }

    // 2) whisper tiny transcription
    let q: Queue = Arc::new(Mutex::new(VecDeque::new()));
    let cancel = Arc::new(AtomicBool::new(false));
    let t = Instant::now();
    let segs = crate::asr::transcribe(&mdir, Kind::Tiny, &samples, "", &q, &cancel)?;
    let dt_asr = t.elapsed().as_secs_f32();
    let joined: String = segs.iter().map(|s| s.text.clone()).collect::<Vec<_>>().join(" ");
    println!(
        "ASR (tiny): {} segments in {:.2}s -> {:?}",
        segs.len(),
        dt_asr,
        joined.trim()
    );
    if joined.trim().chars().count() < 5 {
        bail!("ASR produced no text");
    }

    // 3) translation
    let t = Instant::now();
    let mut tr = crate::translate::Translator::load(&mt)?;
    let ar = tr.translate(joined.trim())?;
    let dt_mt = t.elapsed().as_secs_f32();
    println!("MT: {:.2}s -> {ar}", dt_mt);
    if !ar.chars().any(|c| ('\u{0600}'..='\u{06FF}').contains(&c)) {
        bail!("translation produced no Arabic characters: {ar:?}");
    }

    println!(
        "SELFTEST PASS (total {:.2}s, asr {:.2}s, mt {:.2}s)",
        t_all.elapsed().as_secs_f32(),
        dt_asr,
        dt_mt
    );
    Ok(())
}
