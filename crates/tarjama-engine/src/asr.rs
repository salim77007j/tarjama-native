use anyhow::{Context as _, Result};
use std::path::Path;
use tarjama_core::Kind;

pub struct RawSeg {
    pub t0: i64,
    pub t1: i64,
    pub text: String,
}

/// Transcribe Turkish speech. Returns segments with ms timestamps.
/// Runs inside the engine process; the GUI is protected from any native crash.
pub fn transcribe(
    models_dir: &Path,
    kind: Kind,
    samples: &[f32],
    context: &str,
    n_threads_override: i32,
) -> Result<Vec<RawSeg>> {
    let model_path = models_dir.join(kind.file());
    // GPU engine build (vulkan feature): try the GPU only when our own probe
    // proved a usable Vulkan device exists (whisper.cpp 1.7.1 aborts on a
    // failing vkCreateInstance). TARJAMA_NO_GPU=1 forces CPU for debugging.
    let mut params_ctx = whisper_rs::WhisperContextParameters::default();
    params_ctx.use_gpu = crate::gpu::should_use_gpu();

    let mp = model_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("model path is not valid UTF-8"))?
        .to_string();
    let ctx = whisper_rs::WhisperContext::new_with_params(&mp, params_ctx)
        .with_context(|| format!("failed to load whisper model {}", model_path.display()))?;
    let mut state = ctx.create_state().context("failed to create whisper state")?;

    let cores: i32 = match n_threads_override {
        n if n > 0 => n,
        _ => std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(2)
            .max(1)
            .min(8),
    };

    let mut params = whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::Greedy {
        best_of: 1,
    });
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

    let _hb = crate::pipeline::Heartbeat::start();
    state
        .full(params, samples)
        .context("whisper inference failed")?;
    drop(_hb);

    let n = state.full_n_segments()? as usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t0 = state.full_get_segment_t0(i as i32)?; // centiseconds
        let t1 = state.full_get_segment_t1(i as i32)?;
        let text = state.full_get_segment_text(i as i32)?.trim().to_string();
        if text.is_empty() {
            continue;
        }
        out.push(RawSeg {
            t0: t0 * 10,
            t1: t1 * 10,
            text,
        });
    }
    Ok(out)
}
