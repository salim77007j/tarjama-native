//! GPU (Vulkan) support: runtime device probe + whisper.cpp log capture.
//!
//! The `gpu` engine variant links whisper.cpp with the Vulkan compute backend
//! (GGML_VULKAN=ON). Two jobs live here:
//!
//! 1. `vulkan_probe()` - enumerate Vulkan physical devices through a
//!    dynamically loaded loader (ash). Used by --capabilities so the GUI can
//!    decide whether the GPU engine is worth starting. Fails soft (0 devices)
//!    when the loader is missing or no driver exposes a device.
//! 2. `backend_report()` - whisper.cpp/ggml log lines are captured while the
//!    model loads, so after transcription we can report WHICH backend really
//!    ran (GPU via Vulkan, or CPU after a fallback).

#[cfg(feature = "vulkan")]
use std::sync::Mutex;

#[cfg(feature = "vulkan")]
static CAPTURED: Mutex<Vec<String>> = Mutex::new(Vec::new());
#[cfg(feature = "vulkan")]
struct CaptureLogger;

#[cfg(feature = "vulkan")]
impl log::Log for CaptureLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::Level::Info
    }
    fn log(&self, record: &log::Record) {
        let line = record.args().to_string();
        let interesting = ["Vulkan", "vulkan", "GPU", "backend"]
            .iter()
            .any(|k| line.contains(k));
        if interesting {
            if let Ok(mut g) = CAPTURED.lock() {
                if g.len() < 256 {
                    g.push(line.clone());
                }
            }
        }
        // Mirror to stderr: the GUI shows the stderr tail when a run fails,
        // which makes driver problems diagnosable from the log pane.
        eprintln!("[whisper] {line}");
    }
    fn flush(&self) {}
}
/// Install the log capture once, before any whisper call.
#[cfg(feature = "vulkan")]
pub fn install_log_capture() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let _ = log::set_logger(&CaptureLogger);
        log::set_max_level(log::LevelFilter::Info);
        // Route whisper.cpp + ggml log output into the `log` crate.
        whisper_rs::install_whisper_log_trampoline();
    });
}

/// CPU-only engine builds have no GPU path to report on.
#[cfg(not(feature = "vulkan"))]
pub fn install_log_capture() {}

/// Human-readable report of the compute backend that actually ran, derived
/// from captured whisper.cpp/ggml log lines.
#[cfg(feature = "vulkan")]
pub fn backend_report() -> String {
    let g = CAPTURED.lock().map(|g| g.clone()).unwrap_or_default();
    let used_gpu = g
        .iter()
        .any(|l| l.contains("using Vulkan backend") || l.contains("using CUDA backend"));
    if used_gpu {
        // ggml-vulkan logs: "ggml_vulkan: Found 1 Vulkan devices: Intel(R) ..."
        let mut names: Vec<String> = Vec::new();
        for line in &g {
            if let Some(idx) = line.find("devices:") {
                let rest = line[idx + "devices:".len()..].trim();
                if !rest.is_empty() {
                    names.push(rest.to_string());
                }
            }
        }
        format!(
            "GPU (Vulkan){}",
            if names.is_empty() {
                String::new()
            } else {
                format!(" - {}", names.join("; "))
            }
        )
    } else if g.iter().any(|l| l.contains("Vulkan")) {
        "CPU (Vulkan backend compiled but GPU init failed or no usable device)".to_string()
    } else {
        "CPU".to_string()
    }
}

/// CPU-only engine builds always report the CPU backend.
#[cfg(not(feature = "vulkan"))]
pub fn backend_report() -> String {
    "CPU".to_string()
}

/// Enumerate Vulkan physical devices with a dynamically loaded loader.
/// Returns (device_count, "name1 | name2"). Never panics.
#[cfg(feature = "vulkan")]
pub fn vulkan_probe() -> (u32, String) {
    use ash::vk;

    let entry = match unsafe { ash::Entry::load() } {
        Ok(e) => e,
        Err(_) => return (0, String::new()), // no loader (vulkan-1.dll missing)
    };
    let app = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 1, 0));
    let ci = vk::InstanceCreateInfo::default().application_info(&app);
    let instance = match unsafe { entry.create_instance(&ci, None) } {
        Ok(i) => i,
        Err(_) => return (0, String::new()),
    };
    let devices = unsafe { instance.enumerate_physical_devices() }.unwrap_or_default();
    let mut names = Vec::new();
    for d in &devices {
        let props = unsafe { instance.get_physical_device_properties(*d) };
        let name = unsafe { std::ffi::CStr::from_ptr(props.device_name.as_ptr()) }
            .to_string_lossy()
            .to_string();
        names.push(name);
    }
    let n = devices.len() as u32;
    unsafe { instance.destroy_instance(None) };
    (n, names.join(" | "))
}

#[cfg(not(feature = "vulkan"))]
pub fn vulkan_probe() -> (u32, String) {
    (0, String::new())
}

/// Parts used by caps_line(): (vulkan_compiled, device_count, device_names).
pub fn vulkan_info() -> (u8, u32, String) {
    #[cfg(feature = "vulkan")]
    {
        let (n, names) = vulkan_probe();
        (1, n, names)
    }
    #[cfg(not(feature = "vulkan"))]
    {
        (0, 0, String::new())
    }
}
