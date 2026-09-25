//! Engine subprocess management: spawn, stream JSON events, kill on cancel,
//! and detect native crashes so the GUI can retry with the safe engine.

use anyhow::{anyhow, Result};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use tarjama_core::{EngineEvent, JobSpec};

pub type SharedChild = Arc<Mutex<Option<Child>>>;

pub fn exe_suffix() -> &'static str {
    if cfg!(windows) {
        ".exe"
    } else {
        ""
    }
}

/// Resolve an engine executable. Env overrides (used by tests) first, then
/// files next to the GUI exe: tarjama-engine-fast / tarjama-engine-safe.
pub fn engine_path(variant: &str) -> Option<PathBuf> {
    let env_key = match variant {
        "fast" => "TARJAMA_ENGINE_FAST",
        _ => "TARJAMA_ENGINE_SAFE",
    };
    if let Ok(p) = std::env::var(env_key) {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(d) = exe.parent() {
            let p = d.join(format!("tarjama-engine-{variant}{}", exe_suffix()));
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

/// Pick the engine variant for this CPU. The fast engine is compiled with
/// AVX2+FMA+F16C; the safe engine is plain SSE2 and runs on every x86_64 CPU.
pub fn preferred_variant() -> &'static str {
    if cfg!(target_arch = "x86_64")
        && std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
        && std::arch::is_x86_feature_detected!("f16c")
    {
        "fast"
    } else {
        "safe"
    }
}

pub struct Spawned {
    pub child: SharedChild,
    pub variant: String,
    pub job_file: PathBuf,
}

/// Spawn the engine for a job. Progress is pushed into `queue` as Prog::Engine
/// events; when the process exits exactly one Prog::Exited is pushed.
pub fn spawn_engine(
    job: &JobSpec,
    variant: &str,
    queue: &crate::ui::EventQueue,
) -> Result<Spawned> {
    let path = engine_path(variant).ok_or_else(|| {
        anyhow!(
            "engine not found: tarjama-engine-{variant}{} must sit next to tarjama{}",
            exe_suffix(),
            exe_suffix()
        )
    })?;

    let job_file = std::env::temp_dir().join(format!(
        "tarjama_job_{}_{}.json",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0)
    ));
    std::fs::write(
        &job_file,
        serde_json::to_vec_pretty(job).map_err(|e| anyhow!("job serialize: {e}"))?,
    )?;

    let mut cmd = Command::new(&path);
    cmd.arg("--run")
        .arg(&job_file)
        .env("TARJAMA_ENGINE_VARIANT", variant)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd
        .spawn()
        .map_err(|e| anyhow!("failed to start {}: {e}", path.display()))?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    let handle: SharedChild = Arc::new(Mutex::new(Some(child)));
    let handle2 = handle.clone();
    let queue2 = queue.clone();
    let variant_owned = variant.to_string();
    let job_file2 = job_file.clone();

    std::thread::spawn(move || {
        let mut saw_terminal = false;
        // stdout: JSON protocol lines
        if let Some(out) = stdout {
            let reader = BufReader::new(out);
            for line in reader.lines().map_while(Result::ok) {
                if let Ok(ev) = serde_json::from_str::<EngineEvent>(&line) {
                    match ev {
                        EngineEvent::Done { .. } | EngineEvent::Failed { .. } => {
                            saw_terminal = true;
                        }
                        _ => {}
                    }
                    queue2.lock().unwrap().push_back(crate::ui::Prog::Engine(ev));
                } else {
                    queue2
                        .lock()
                        .unwrap()
                        .push_back(crate::ui::Prog::Engine(EngineEvent::Log { s: line }));
                }
            }
        }
        // stderr: native crash chatter / panic text
        let mut err_tail = String::new();
        if let Some(mut e) = stderr {
            use std::io::Read;
            let mut buf = String::new();
            let _ = e.read_to_string(&mut buf);
            err_tail = buf.lines().rev().take(4).collect::<Vec<_>>().join(" | ");
            if err_tail.len() > 300 {
                err_tail.truncate(300);
            }
        }
        let code = {
            let mut guard = handle2.lock().unwrap();
            match guard.as_mut() {
                Some(c) => c.wait().ok().and_then(|s| s.code()),
                None => None, // killed by Cancel (child already taken/killed)
            }
        };
        let _ = std::fs::remove_file(&job_file2);
        queue2.lock().unwrap().push_back(crate::ui::Prog::Exited {
            variant: variant_owned,
            code,
            stderr: err_tail,
            saw_terminal,
        });
    });

    Ok(Spawned {
        child: handle,
        variant: variant.to_string(),
        job_file,
    })
}

/// Kill a running engine (real Cancel).
pub fn kill(child: &SharedChild) {
    if let Ok(mut g) = child.lock() {
        if let Some(c) = g.as_mut() {
            let _ = c.kill();
        }
    }
}

/// Forward --selftest / --cli to the safe engine; returns its exit code.
pub fn forward_to_engine(args: &[String]) -> i32 {
    let Some(path) = engine_path("safe").or_else(|| engine_path("fast")) else {
        eprintln!(
            "tarjama engine not found next to the app (tarjama-engine-safe{})",
            exe_suffix()
        );
        return 2;
    };
    match Command::new(path).args(args).status() {
        Ok(s) => s.code().unwrap_or(1),
        Err(e) => {
            eprintln!("failed to run engine: {e}");
            1
        }
    }
}
