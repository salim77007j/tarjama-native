use anyhow::{anyhow, bail, Context as _, Result};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
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
}

/// One timed subtitle line (times in milliseconds).
#[derive(Clone, Debug)]
pub struct Seg {
    pub t0: i64,
    pub t1: i64,
    pub tr: String,
    pub ar: String,
}

/// Progress messages flowing from the worker thread to the UI/CLI.
pub enum Prog {
    Stage(String),
    Asr(f32),
    Mt(usize, usize),
    Log(String),
    Done(Vec<Seg>),
    Failed(String),
}

pub type Queue = Arc<Mutex<std::collections::VecDeque<Prog>>>;

pub fn push(q: &Queue, p: Prog) {
    if let Ok(mut g) = q.lock() {
        g.push_back(p);
    }
}

pub struct Params {
    pub input: PathBuf,
    pub model: Kind,
    pub context: String,
    pub glossary: String,
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

pub fn run(p: Params, q: Queue, cancel: Arc<AtomicBool>) {
    match run_inner(&p, &q, &cancel) {
        Ok(segs) => push(&q, Prog::Done(segs)),
        Err(e) => push(&q, Prog::Failed(format!("{e:#}"))),
    }
}

fn run_inner(p: &Params, q: &Queue, cancel: &AtomicBool) -> Result<Vec<Seg>> {
    let mdir = models_dir()?;
    push(
        q,
        Prog::Log(format!("Input: {} ({})", p.input.display(), file_size(&p.input))),
    );

    // 1) Extract audio (mono 16 kHz float32) with bundled ffmpeg
    push(q, Prog::Stage("Extracting audio...".into()));
    let samples =
        crate::audio::extract(&p.input, cancel).context("Audio extraction failed (ffmpeg missing or file unreadable?)")?;
    if cancel.load(Ordering::Relaxed) {
        bail!("Cancelled");
    }
    let secs = samples.len() as f32 / 16000.0;
    push(
        q,
        Prog::Log(format!("Audio: {:.1}s, {} samples (mono 16 kHz)", secs, samples.len())),
    );
    if samples.len() < 1600 {
        bail!("No usable audio track found in this file");
    }

    // 2) Transcribe Turkish
    push(q, Prog::Stage(format!("Transcribing ({:?})...", p.model)));
    let raw = crate::asr::transcribe(&mdir, p.model, &samples, &p.context, q, cancel)?;
    if cancel.load(Ordering::Relaxed) {
        bail!("Cancelled");
    }
    if raw.is_empty() {
        bail!("No speech detected in this file");
    }
    push(q, Prog::Log(format!("{} speech segments found", raw.len())));

    // 3) Translate to Arabic
    push(q, Prog::Stage("Translating to Arabic...".into()));
    let mut tr = crate::translate::Translator::load(&mdir.join("opus-mt"))
        .context("Failed to load translation model (models/opus-mt)")?;
    let n = raw.len();
    let mut segs = Vec::with_capacity(n);
    for (i, r) in raw.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            bail!("Cancelled");
        }
        let ar = tr.translate(&r.text).unwrap_or_default();
        segs.push(Seg { t0: r.t0, t1: r.t1, tr: r.text.clone(), ar });
        push(q, Prog::Mt(i + 1, n));
    }

    // 4) Apply user glossary (tr=ar pairs)
    let pairs = crate::glossary::parse(&p.glossary);
    if !pairs.is_empty() {
        push(q, Prog::Log(format!("Applying glossary: {} pairs", pairs.len())));
        for s in &mut segs {
            crate::glossary::apply(&mut s.ar, &pairs);
        }
    }

    Ok(segs)
}

fn file_size(p: &Path) -> String {
    let b = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
    if b >= 1_048_576 {
        format!("{:.1} MB", b as f64 / 1_048_576.0)
    } else {
        format!("{:.0} KB", b as f64 / 1024.0)
    }
}
