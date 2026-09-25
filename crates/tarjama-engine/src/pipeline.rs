//! Full TR->AR pipeline: ffmpeg audio extraction -> whisper ASR -> ONNX MT.
//! All progress is streamed as JSON events with immediate flush.

use anyhow::{bail, Context as _, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tarjama_core::{JobSpec, Seg};

pub fn run(job: &JobSpec) -> Result<()> {
    let segs = run_collect(job)?;
    tarjama_core::emit(&tarjama_core::EngineEvent::Done { segs });
    Ok(())
}

pub fn run_collect(job: &JobSpec) -> Result<Vec<Seg>> {
    let t_start = Instant::now();
    let mdir = &job.models_dir;
    log(&format!(
        "Input: {} ({})",
        job.input.display(),
        tarjama_core::file_size(&job.input)
    ));

    // 1) Extract audio (mono 16 kHz float32) with ffmpeg
    emit_stage("extracting");
    let samples = crate::audio::extract(&job.input, &job.ffmpeg)
        .context("Audio extraction failed (is the file a valid video/audio?)")?;
    let secs = samples.len() as f32 / 16000.0;
    log(&format!(
        "Audio: {:.1}s, {} samples (mono 16 kHz)",
        secs,
        samples.len()
    ));
    if samples.len() < 1600 {
        bail!("No usable audio track found in this file");
    }

    // 2) Transcribe Turkish
    emit_stage("loading-model");
    let model_path = mdir.join(job.model.file());
    let size = tarjama_core::file_size(&model_path);
    log(&format!(
        "Loading whisper model {} ({})...",
        model_path.display(),
        size
    ));
    let t_load = Instant::now();
    emit_stage("transcribe");
    let raw = crate::asr::transcribe(mdir, job.model, &samples, &job.context, job.n_threads)?;
    log(&format!(
        "ASR done in {:.1}s, {} speech segments (first: {:?}) | compute backend: {}",
        t_load.elapsed().as_secs_f32(),
        raw.len(),
        raw.first().map(|s| s.text.as_str()).unwrap_or(""),
        crate::gpu::backend_report()
    ));
    if raw.is_empty() {
        bail!("No speech detected in this file");
    }

    // 3) Translate to Arabic (OPUS-MT int8, beam search, v1.3)
    emit_stage("loading-mt");
    let mut tr = crate::translate::Translator::load(mdir)
        .context("Failed to load translation model (models/mt/tr-ar)")?;
    emit_stage("translate");
    let n = raw.len();
    let mut segs = Vec::with_capacity(n);
    for (i, r) in raw.iter().enumerate() {
        // whisper (especially tiny/base) sometimes wraps segments in stray
        // quotation marks; they are noise for MT and for subtitles.
        let clean: String = r
            .text
            .replace(['"', '\u{201C}', '\u{201D}'], "")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        let ar = tr.translate(&clean).unwrap_or_default();
        segs.push(Seg {
            t0: r.t0,
            t1: r.t1,
            tr: clean,
            ar,
        });
        tarjama_core::emit(&tarjama_core::EngineEvent::Mt { a: i + 1, b: n });
    }

    // 4) Apply user glossary (tr=ar pairs)
    let pairs = tarjama_core::glossary::parse(&job.glossary);
    if !pairs.is_empty() {
        log(&format!("Applying glossary: {} pairs", pairs.len()));
        for s in &mut segs {
            tarjama_core::glossary::apply(&mut s.ar, &pairs);
        }
    }

    log(&format!(
        "Pipeline finished in {:.1}s",
        t_start.elapsed().as_secs_f32()
    ));
    Ok(segs)
}

fn emit_stage(s: &str) {
    tarjama_core::emit(&tarjama_core::EngineEvent::Stage { s: s.into() });
}

fn log(s: &str) {
    tarjama_core::emit(&tarjama_core::EngineEvent::Log { s: s.into() });
}

/// Heartbeat: proves the engine is alive while whisper runs (blocking call).
pub struct Heartbeat {
    stop: Arc<AtomicBool>,
}

impl Heartbeat {
    pub fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop2 = stop.clone();
        std::thread::spawn(move || {
            let t0 = Instant::now();
            while !stop2.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_secs(5));
                if stop2.load(Ordering::Relaxed) {
                    break;
                }
                tarjama_core::emit(&tarjama_core::EngineEvent::Heartbeat {
                    s: t0.elapsed().as_secs(),
                });
            }
        });
        Self { stop }
    }
}

impl Drop for Heartbeat {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}
