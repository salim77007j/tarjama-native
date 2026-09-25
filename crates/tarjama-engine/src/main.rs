//! tarjama-engine: offline Turkish ASR + Arabic translation pipeline.
//!
//! Runs as a SEPARATE PROCESS spawned by the GUI so that a native crash inside
//! whisper.cpp (or anywhere in the C stack) can never take the UI down. All
//! progress flows as one JSON event per line on stdout (tarjama_core::EngineEvent).
//!
//! Modes:
//!   tarjama-engine --run job.json     full pipeline for the GUI
//!   tarjama-engine --cli --input F    human-oriented CLI (writes subtitle files)
//!   tarjama-engine --selftest         verify models + full pipeline on embedded sample
//!   tarjama-engine --capabilities     print compiled SIMD + Vulkan flags (CI assertion)

mod asr;
mod audio;
mod gpu;
mod pipeline;
mod translate;

use anyhow::{bail, Result};
use std::time::Instant;
use tarjama_core::{Kind, APP_VERSION};

// Compile-time SIMD flags of the linked whisper.cpp (ggml). These reflect what
// was compiled into THIS binary, not the runtime CPU - the CI asserts on them
// so a mis-flagged build can never be shipped again.
extern "C" {
    fn ggml_cpu_has_sse3() -> i32;
    fn ggml_cpu_has_avx() -> i32;
    fn ggml_cpu_has_avx2() -> i32;
    fn ggml_cpu_has_fma() -> i32;
    fn ggml_cpu_has_f16c() -> i32;
    fn ggml_cpu_has_avx512() -> i32;
}

pub struct Caps {
    pub sse3: i32,
    pub avx: i32,
    pub avx2: i32,
    pub fma: i32,
    pub f16c: i32,
    pub avx512: i32,
}

pub fn caps() -> Caps {
    unsafe {
        Caps {
            sse3: ggml_cpu_has_sse3(),
            avx: ggml_cpu_has_avx(),
            avx2: ggml_cpu_has_avx2(),
            fma: ggml_cpu_has_fma(),
            f16c: ggml_cpu_has_f16c(),
            avx512: ggml_cpu_has_avx512(),
        }
    }
}

pub fn caps_line() -> String {
    let c = caps();
    let (vk, vk_devs, vk_names) = gpu::vulkan_info();
    format!(
        "engine v{APP_VERSION} compiled-flags: sse3={} avx={} avx2={} fma={} f16c={} avx512={} vulkan={} vulkan-devices={} vulkan-gpus={}",
        c.sse3,
        c.avx,
        c.avx2,
        c.fma,
        c.f16c,
        c.avx512,
        vk,
        vk_devs,
        if vk_names.is_empty() { "none".to_string() } else { vk_names }
    )
}

/// Standard engine banner as the very first protocol line.
pub fn announce(variant: &str) {
    tarjama_core::emit(&tarjama_core::EngineEvent::Log {
        s: format!(
            "Tarjama engine v{APP_VERSION} ({variant}) | {}",
            caps_line()
        ),
    });
}

fn main() {
    if let Err(e) = real_main() {
        // Emit the failure as a protocol event first (GUI shows it), then exit.
        tarjama_core::emit(&tarjama_core::EngineEvent::Failed {
            err: format!("{e:#}"),
        });
        eprintln!("tarjama-engine error: {e:#}");
        std::process::exit(1);
    }
}

fn real_main() -> Result<()> {
    // Capture whisper.cpp/ggml logs (backend detection) before anything runs.
    gpu::install_log_capture();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        bail!("usage: tarjama-engine (--run job.json | --cli ... | --selftest | --capabilities)");
    }
    match args[0].as_str() {
        "--capabilities" => {
            println!("{}", caps_line());
            Ok(())
        }
        "--mt-debug" => {
            // Debug: translate the given text. Usage: tarjama-engine --mt-debug "metin"
            announce("mt-debug");
            let text = args.get(1).ok_or_else(|| anyhow::anyhow!("--mt-debug requires text"))?;
            let mdir = tarjama_core::models_dir()?;
            let mut tr = translate::Translator::load(&mdir)?;
            let ar = tr.translate(text)?;
            println!("AR: {ar}");
            Ok(())
        }
        "--selftest" => {
            announce("selftest");
            selftest()
        }
        "--run" => {
            let path = args
                .get(1)
                .ok_or_else(|| anyhow::anyhow!("--run requires a job.json path"))?;
            let raw = std::fs::read_to_string(path)?;
            let job: tarjama_core::JobSpec = serde_json::from_str(&raw)?;
            let variant = std::env::var("TARJAMA_ENGINE_VARIANT").unwrap_or_default();
            announce(&variant);
            pipeline::run(&job)
        }
        "--cli" => cli(&args[1..]),
        other => bail!("unknown mode '{other}'"),
    }
}

const SELFTEST_WAV: &[u8] = include_bytes!("../assets/selftest_tr.wav");

fn selftest() -> Result<()> {
    println!("Tarjama Studio Native - engine selftest");
    println!("{}", caps_line());
    println!("=======================================");
    let t_all = Instant::now();
    let require_all = std::env::var("TARJAMA_SELFTEST_REQUIRE").as_deref() == Ok("all");

    let mdir = tarjama_core::models_dir()?;
    println!("models dir: {}", mdir.display());

    for k in Kind::all() {
        let p = mdir.join(k.file());
        match std::fs::metadata(&p) {
            Ok(meta) => println!(
                "  ok {:?} {} ({:.1} MB)",
                k,
                p.display(),
                meta.len() as f64 / 1e6
            ),
            Err(_) => {
                if require_all {
                    anyhow::bail!("MISSING model file: {}", p.display());
                }
                println!("  -- {:?} not present (skipped)", k);
            }
        }
    }
    let mt_ok = {
        let p = mdir.join("mt").join("tr-ar");
        ["encoder_model.int8.onnx", "decoder_model.int8.onnx", "decoder_with_past_model.int8.onnx", "tokenizer.json", "meta.json"]
            .iter()
            .all(|f| p.join(f).is_file())
    };
    if !mt_ok {
        anyhow::bail!(
            "MISSING translation model: need {}/mt/tr-ar/* (encoder + decoder + decoder_with_past int8, tokenizer, meta)",
            mdir.display()
        );
    }
    println!("  ok mt/tr-ar (OPUS-MT int8, beam search x{})", 4);

    let samples = tarjama_core::decode_wav(SELFTEST_WAV)?;
    println!(
        "sample: {} samples ({:.1}s)",
        samples.len(),
        samples.len() as f64 / 16000.0
    );
    if samples.len() < 8000 {
        bail!("selftest wav too small");
    }

    let t = Instant::now();
    let segs = asr::transcribe(&mdir, Kind::Tiny, &samples, "", 0)?;
    let dt_asr = t.elapsed().as_secs_f32();
    println!("compute backend: {}", gpu::backend_report());
    let joined: String = segs
        .iter()
        .map(|s| s.text.clone())
        .collect::<Vec<_>>()
        .join(" ");
    println!(
        "ASR (tiny): {} segments in {:.2}s -> {:?}",
        segs.len(),
        dt_asr,
        joined.trim()
    );
    if joined.trim().chars().count() < 5 {
        bail!("ASR produced no text");
    }

    let t = Instant::now();
    let mut tr = translate::Translator::load(&mdir)?;
    let ar = tr.translate(joined.trim())?;
    let dt_mt = t.elapsed().as_secs_f32();
    println!("MT (beam 4): {:.2}s -> {ar}", dt_mt);
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

/// Human-oriented CLI (writes subtitle files, prints plain text).
fn cli(args: &[String]) -> Result<()> {
    let mut input: Option<std::path::PathBuf> = None;
    let mut model = Kind::Small;
    let mut out: Option<std::path::PathBuf> = None;
    let mut context = String::new();
    let mut glossary = String::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => {
                i += 1;
                input = Some(std::path::PathBuf::from(&args[i]));
            }
            "--model" => {
                i += 1;
                model = Kind::parse(&args[i])?;
            }
            "--out" => {
                i += 1;
                out = Some(std::path::PathBuf::from(&args[i]));
            }
            "--context" => {
                i += 1;
                context = args[i].clone();
            }
            "--glossary" => {
                i += 1;
                glossary = std::fs::read_to_string(&args[i])?;
            }
            other => bail!("unknown cli arg '{other}'"),
        }
        i += 1;
    }
    let input = input.ok_or_else(|| anyhow::anyhow!("--input required"))?;
    if !input.is_file() {
        bail!("input file not found: {}", input.display());
    }
    let out = out.unwrap_or_else(|| input.with_extension(""));
    if let Some(d) = out.parent() {
        if !d.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(d);
        }
    }
    let mdir = tarjama_core::models_dir()?;

    let job = tarjama_core::JobSpec {
        input: input.clone(),
        model,
        context,
        glossary,
        models_dir: mdir,
        ffmpeg: String::new(),
        out_prefix: Some(out.clone()),
        n_threads: 0,
    };

    let segs = pipeline::run_collect(&job)?;
    println!(
        "{} segments in {:.1}s",
        segs.len(),
        Instant::now().elapsed().as_secs_f32()
    );
    for (suffix, body) in tarjama_core::export_strings(&segs) {
        let p = std::path::PathBuf::from(format!(
            "{}.{}",
            out.display(),
            suffix
        ));
        std::fs::write(&p, body)?;
        println!("wrote {}", p.display());
    }
    Ok(())
}
