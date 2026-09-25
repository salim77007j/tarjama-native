use crate::pipeline::{Kind, Queue};
use anyhow::{Context as _, Result};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

pub struct RawSeg {
    pub t0: i64,
    pub t1: i64,
    pub text: String,
}

/// Transcribe Turkish speech. Returns segments with ms timestamps.
pub fn transcribe(
    models_dir: &Path,
    kind: Kind,
    samples: &[f32],
    context: &str,
    q: &Queue,
    cancel: &AtomicBool,
) -> Result<Vec<RawSeg>> {
    let model_path = models_dir.join(kind.file());
    let params_ctx = whisper_rs::WhisperContextParameters::default();

    let mp = model_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("model path is not valid UTF-8"))?
        .to_string();
    let ctx = whisper_rs::WhisperContext::new_with_params(&mp, params_ctx)
        .with_context(|| format!("failed to load whisper model {}", model_path.display()))?;
    let mut state = ctx.create_state().context("failed to create whisper state")?;

    let cores: i32 = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(2)
        .max(1);

    let mut params = whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::Greedy { best_of: 5 });
    params.set_language(Some("tr"));
    params.set_translate(false);
    params.set_n_threads(cores);
    params.set_print_progress(false);
    params.set_print_special(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    let ctx_prompt = context.trim().to_string();
    if !ctx_prompt.is_empty() {
        params.set_initial_prompt(ctx_prompt.as_str());
    }
    // NOTE: set_progress_callback_safe segfaults in whisper-rs 0.13.2 during
    // full() on some platforms, so progress is stage-based instead of a
    // percentage. Do not re-enable without testing.
    let _ = q;

    state
        .full(params, samples)
        .context("whisper inference failed")?;

    let n = state.full_n_segments()? as usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let t0 = state.full_get_segment_t0(i as i32)?; // centiseconds
        let t1 = state.full_get_segment_t1(i as i32)?;
        let text = state.full_get_segment_text(i as i32)?.trim().to_string();
        if text.is_empty() {
            continue;
        }
        out.push(RawSeg { t0: t0 * 10, t1: t1 * 10, text });
    }
    Ok(out)
}
