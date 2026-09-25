use anyhow::{anyhow, bail, Context as _, Result};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};

/// Locate ffmpeg: env TARJAMA_FFMPEG > next to exe > models dir > PATH.
pub fn ffmpeg_path() -> Result<PathBuf> {
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

/// Extract mono 16 kHz f32 samples from any media file via ffmpeg.
pub fn extract(path: &Path, cancel: &AtomicBool) -> Result<Vec<f32>> {
    let ff = ffmpeg_path()?;
    let mut child = Command::new(&ff)
        .args([
            "-nostdin",
            "-v",
            "error",
            "-i",
        ])
        .arg(path)
        .args(["-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("failed to launch ffmpeg at {}", ff.display()))?;

    let mut out = Vec::new();
    {
        let mut stdout = child.stdout.take().context("no ffmpeg stdout")?;
        let mut buf = [0u8; 1 << 16];
        loop {
            if cancel.load(Ordering::Relaxed) {
                let _ = child.kill();
                bail!("Cancelled");
            }
            let n = stdout.read(&mut buf)?;
            if n == 0 {
                break;
            }
            out.extend_from_slice(&buf[..n]);
        }
    }
    let status = child.wait()?;
    if !status.success() && out.is_empty() {
        return Err(anyhow!(
            "ffmpeg failed (exit {:?}). Unsupported file or missing audio stream.",
            status.code()
        ));
    }
    if out.len() % 4 != 0 {
        out.truncate(out.len() - out.len() % 4);
    }
    let samples: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(samples)
}
