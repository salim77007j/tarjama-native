use crate::pipeline::{self, Kind, Params, Prog};
use anyhow::{bail, Result};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Headless mode: tarjama --cli --input FILE [--model tiny|base|small] [--out PREFIX]
/// [--context "..."] [--glossary FILE] [--formats srt,bilingual,vtt,txt]
pub fn run(args: &[String]) -> Result<()> {
    let mut input: Option<PathBuf> = None;
    let mut model = Kind::Small;
    let mut out: Option<PathBuf> = None;
    let mut context = String::new();
    let mut glossary = String::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => {
                i += 1;
                input = Some(PathBuf::from(&args[i]));
            }
            "--model" => {
                i += 1;
                model = match args[i].as_str() {
                    "tiny" => Kind::Tiny,
                    "base" => Kind::Base,
                    "small" => Kind::Small,
                    other => bail!("unknown model '{other}' (tiny|base|small)"),
                };
            }
            "--out" => {
                i += 1;
                out = Some(PathBuf::from(&args[i]));
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

    let q: Arc<Mutex<VecDeque<Prog>>> = Arc::new(Mutex::new(VecDeque::new()));
    let cancel = Arc::new(AtomicBool::new(false));
    let t = Instant::now();
    pipeline::run(
        Params { input, model, context, glossary },
        q.clone(),
        cancel,
    );
    let mut segs = Vec::new();
    loop {
        let msgs: Vec<Prog> = {
            let mut g = q.lock().unwrap();
            g.drain(..).collect()
        };
        for m in msgs {
            match m {
                Prog::Stage(s) => println!("[stage] {s}"),
                Prog::Asr(p) => print!("\r[asr] {:>3}%", (p * 100.0) as i32),
                Prog::Mt(a, b) => print!("\r[mt ] {a}/{b}"),
                Prog::Log(s) => println!("{s}"),
                Prog::Done(s) => segs = s,
                Prog::Failed(e) => bail!("{e}"),
            }
        }
        if !segs.is_empty() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    println!("\n{} segments in {:.1}s", segs.len(), t.elapsed().as_secs_f32());

    std::fs::write(format!("{}.ar.srt", out.display()), crate::srt::srt_ar(&segs))?;
    std::fs::write(format!("{}.bilingual.srt", out.display()), crate::srt::srt_bilingual(&segs))?;
    std::fs::write(format!("{}.ar.vtt", out.display()), crate::srt::vtt_ar(&segs))?;
    std::fs::write(format!("{}.ar.txt", out.display()), crate::srt::txt_ar(&segs))?;
    std::fs::write(format!("{}.tr.txt", out.display()), crate::srt::txt_tr(&segs))?;
    println!(
        "wrote {}.ar.srt, {}.bilingual.srt, {}.ar.vtt, {}.ar.txt, {}.tr.txt",
        out.display(),
        out.display(),
        out.display(),
        out.display(),
        out.display()
    );
    Ok(())
}
