use anyhow::{anyhow, bail, Context as _, Result};
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use tarjama_core::ffmpeg_path;

/// Extract mono 16 kHz f32 samples from any media file via ffmpeg.
/// ffmpeg's stderr is drained on a separate thread (reading it inline would
/// deadlock: blocked stderr read vs. blocked stdout write).
pub fn extract(path: &Path, ffmpeg: &str) -> Result<Vec<f32>> {
    let ff = ffmpeg_path(ffmpeg)?;
    let mut child = Command::new(&ff)
        .args(["-nostdin", "-v", "error", "-i"])
        .arg(path)
        .args(["-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to launch ffmpeg at {}", ff.display()))?;

    let mut stderr_text = String::new();
    let stderr_thread: Option<std::thread::JoinHandle<String>> = child.stderr.take().map(|mut e| {
        std::thread::spawn(move || {
            use std::io::Read;
            let mut buf = String::new();
            let _ = e.read_to_string(&mut buf);
            buf
        })
    });

    let mut out = Vec::new();
    {
        let mut stdout = child.stdout.take().context("no ffmpeg stdout")?;
        let mut buf = [0u8; 1 << 16];
        loop {
            let n = stdout.read(&mut buf)?;
            if n == 0 {
                break;
            }
            out.extend_from_slice(&buf[..n]);
        }
    }
    let status = child.wait()?;
    if let Some(h) = stderr_thread {
        match h.join() {
            Ok(s) => stderr_text = s,
            Err(_) => {}
        }
    }
    if !status.success() {
        let detail = stderr_text.lines().last().unwrap_or("").to_string();
        return Err(anyhow!(
            "ffmpeg failed (exit {:?}){}{}",
            status.code(),
            if detail.is_empty() { "" } else { ": " },
            detail
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
