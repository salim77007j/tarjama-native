//! Shared types for Tarjama Studio Native.
//!
//! The engine is a separate process so a native crash inside whisper.cpp can
//! never take the GUI down. This crate defines everything both sides agree on:
//! model kinds, subtitle segments, the JSON event protocol and job spec.

use anyhow::{anyhow, Context as _, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub mod glossary;
pub mod srt;

pub const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Bundled whisper models. Tiny / Base / Small ONLY - medium and large-v3
/// turbo are intentionally not part of this application.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum Kind {
    Tiny,
    Base,
    Small,
}

impl Kind {
    pub fn all() -> [Kind; 3] {
        [Kind::Tiny, Kind::Base, Kind::Small]
    }

    pub fn file(self) -> &'static str {
        match self {
            Kind::Tiny => "ggml-tiny-q5_1.bin",
            Kind::Base => "ggml-base-q5_1.bin",
            Kind::Small => "ggml-small-q5_1.bin",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Kind::Tiny => "Tiny  (32 MB - fastest)",
            Kind::Base => "Base  (60 MB - balanced)",
            Kind::Small => "Small (190 MB - best quality)",
        }
    }

    pub fn parse(s: &str) -> Result<Kind> {
        match s.to_ascii_lowercase().as_str() {
            "tiny" => Ok(Kind::Tiny),
            "base" => Ok(Kind::Base),
            "small" => Ok(Kind::Small),
            other => Err(anyhow!("unknown model '{other}' (tiny|base|small)")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Tiny => "tiny",
            Kind::Base => "base",
            Kind::Small => "small",
        }
    }
}

/// One timed subtitle line (times in milliseconds).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Seg {
    pub t0: i64,
    pub t1: i64,
    pub tr: String,
    pub ar: String,
}

/// Event protocol: engine prints one JSON object per line on stdout.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "t", rename_all = "snake_case")]
pub enum EngineEvent {
    /// Free-form log line for the UI log pane.
    Log { s: String },
    /// Named pipeline stage (extract / load-model / transcribe / translate).
    Stage { s: String },
    /// Translation progress: a of b lines.
    Mt { a: usize, b: usize },
    /// Engine still alive while whisper runs (seconds elapsed).
    Heartbeat { s: u64 },
    /// Pipeline finished successfully.
    Done { segs: Vec<Seg> },
    /// Pipeline failed with a readable message (engine exits 0 after this).
    Failed { err: String },
}

/// Print one protocol event to stdout and flush immediately.
pub fn emit(ev: &EngineEvent) {
    if let Ok(line) = serde_json::to_string(ev) {
        use std::io::Write;
        let mut out = std::io::stdout().lock();
        let _ = writeln!(out, "{line}");
        let _ = out.flush();
    }
}

/// A full translation job, passed to the engine as JSON via --run <file>.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JobSpec {
    pub input: PathBuf,
    pub model: Kind,
    #[serde(default)]
    pub context: String,
    #[serde(default)]
    pub glossary: String,
    pub models_dir: PathBuf,
    /// Explicit ffmpeg path; empty = resolve automatically.
    #[serde(default)]
    pub ffmpeg: String,
    /// When set, the engine also writes subtitle files with this prefix.
    #[serde(default)]
    pub out_prefix: Option<PathBuf>,
    /// 0 = automatic thread count.
    #[serde(default)]
    pub n_threads: i32,
}

/// Resolve the directory containing bundled models (ggml-*.bin + opus-mt/).
pub fn models_dir() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("TARJAMA_MODELS_DIR") {
        let p = PathBuf::from(p);
        if p.join("opus-mt").is_dir() {
            return Ok(p);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(d) = exe.parent() {
            let p = d.join("models");
            if p.join("opus-mt").is_dir() {
                return Ok(p);
            }
            if let Some(d2) = d.parent() {
                let p2 = d2.join("models");
                if p2.join("opus-mt").is_dir() {
                    return Ok(p2);
                }
                if let Some(d3) = d2.parent() {
                    let p3 = d3.join("models");
                    if p3.join("opus-mt").is_dir() {
                        return Ok(p3);
                    }
                }
            }
        }
    }
    let p = PathBuf::from("models");
    if p.join("opus-mt").is_dir() {
        return Ok(p);
    }
    Err(anyhow!(
        "models folder not found: it must sit next to tarjama.exe (models/ggml-*.bin + models/opus-mt/)"
    ))
}

pub fn file_size(p: &Path) -> String {
    let b = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
    if b >= 1_048_576 {
        format!("{:.1} MB", b as f64 / 1_048_576.0)
    } else {
        format!("{:.0} KB", b as f64 / 1024.0)
    }
}

/// Minimal RIFF WAV parser (16-bit PCM -> f32 mono 16 kHz).
pub fn decode_wav(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        anyhow::bail!("not a RIFF WAVE file");
    }
    let mut pos = 12usize;
    let mut fmt: Option<(u16, u16, u32, u16)> = None;
    let mut data: Option<&[u8]> = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size =
            u32::from_le_bytes([bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]])
                as usize;
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
        anyhow::bail!("only 16-bit PCM supported (fmt={fmt}, bits={bits})");
    }
    let mut samples: Vec<f32> = raw
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
        .collect();
    if channels > 1 {
        samples = samples
            .chunks(channels as usize)
            .map(|c| c.iter().sum::<f32>() / c.len() as f32)
            .collect();
    }
    if rate != 16000 {
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

/// Collect exported subtitle strings for one job (used by engine --run and tests).
pub fn export_strings(segs: &[Seg]) -> Vec<(String, String)> {
    vec![
        ("ar.srt".into(), srt::srt_ar(segs)),
        ("bilingual.srt".into(), srt::srt_bilingual(segs)),
        ("ar.vtt".into(), srt::vtt_ar(segs)),
        ("ar.txt".into(), srt::txt_ar(segs)),
        ("tr.txt".into(), srt::txt_tr(segs)),
    ]
}

/// Locate ffmpeg: explicit path > env TARJAMA_FFMPEG > next to exe > PATH.
pub fn ffmpeg_path(explicit: &str) -> Result<PathBuf> {
    if !explicit.trim().is_empty() {
        let p = PathBuf::from(explicit);
        if p.is_file() {
            return Ok(p);
        }
        return Err(anyhow!("ffmpeg not found at {}", p.display()));
    }
    if let Ok(p) = std::env::var("TARJAMA_FFMPEG") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Ok(p);
        }
    }
    let exe_name = if cfg!(windows) { "ffmpeg.exe" } else { "ffmpeg" };
    if let Ok(exe) = std::env::current_exe() {
        if let Some(d) = exe.parent() {
            let p = d.join(exe_name);
            if p.is_file() {
                return Ok(p);
            }
            let p2 = d.join("bin").join(exe_name);
            if p2.is_file() {
                return Ok(p2);
            }
        }
    }
    Ok(PathBuf::from(exe_name))
}

/// Helper used by engine error paths: run a context closure and wrap errors.
pub fn wrap_err<T>(r: Result<T>, msg: &str) -> Result<T> {
    r.with_context(|| msg.to_string())
}
