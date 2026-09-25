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
/// files next to the GUI exe: tarjama-engine-gpu / -fast / -safe.
pub fn engine_path(variant: &str) -> Option<PathBuf> {
    let env_key = match variant {
        "gpu" => "TARJAMA_ENGINE_GPU",
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

/// Downgrade chain used whenever a variant fails to start or crashes.
pub fn next_variant(variant: &str) -> Option<&'static str> {
    match variant {
        "gpu" => Some("fast"),
        "fast" => Some("safe"),
        _ => None,
    }
}

/// Probe the GPU engine's --capabilities output. Returns (vulkan_devices,
/// gpu_names). Cached for the lifetime of the GUI process. The probe is
/// time-boxed: a missing Vulkan loader, a hung driver or a broken binary all
/// simply yield None so CPU engines stay available.
fn probe_gpu_caps() -> Option<(u32, String)> {
    static CACHE: std::sync::OnceLock<Option<(u32, String)>> = std::sync::OnceLock::new();
    CACHE
        .get_or_init(|| {
            let path = engine_path("gpu")?;
            let mut child = Command::new(&path)
                .arg("--capabilities")
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()
                .ok()?; // missing vulkan-1.dll etc. -> spawn error -> no GPU
            let (tx, rx) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                let out = child.wait_with_output();
                let _ = tx.send(out.ok().map(|o| o.stdout));
            });
            let stdout = match rx.recv_timeout(std::time::Duration::from_secs(4)) {
                Ok(Some(s)) => s,
                _ => return None, // timeout or wait error
            };
            let s = String::from_utf8_lossy(&stdout).to_string();
            let devs = s
                .split_whitespace()
                .find_map(|t| t.strip_prefix("vulkan-devices="))
                .and_then(|v| v.parse::<u32>().ok())?;
            let names = s
                .split_whitespace()
                .find(|t| t.starts_with("vulkan-gpus="))
                .map(|t| t.trim_start_matches("vulkan-gpus=").to_string())
                .unwrap_or_default();
            Some((devs, names))
        })
        .clone()
}

/// Pick the engine variant for this machine, best first:
///   1. gpu  - whisper.cpp on the GPU via Vulkan (any vendor, incl. iGPUs)
///   2. fast - AVX2+FMA+F16C CPU build
///   3. safe - plain SSE2, runs on every x86_64 CPU
pub fn preferred_variant() -> &'static str {
    if let Some((devs, _)) = probe_gpu_caps() {
        if devs >= 1 {
            return "gpu";
        }
    }
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

/// Forward --selftest / --cli to the best available engine; returns its exit code.
pub fn forward_to_engine(args: &[String]) -> i32 {
    let Some(path) = engine_path("safe")
        .or_else(|| engine_path("fast"))
        .or_else(|| engine_path("gpu"))
    else {
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
