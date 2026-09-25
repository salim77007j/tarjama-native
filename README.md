# Tarjama Studio Native

Standalone, fully offline desktop app: **Turkish video/audio -> Arabic subtitles**.
Pure native Rust (no web technologies, no browser, no installer, no internet).

## v1.1 - reliability overhaul

The v1.0 build crashed at "Transcribing..." on CPUs without AVX2/AVX512: the C
build silently baked the GitHub runner's CPU features into the exe (the
`WHISPER_NO_*` env vars were ignored by whisper-rs-sys 0.11). v1.1 fixes this
properly, in three independent layers:

1. **Correct builds** - `vendor/whisper-rs-sys` is a patched fork that forwards
   `TARJAMA_GGML_*` env vars as real CMake cache variables. CI builds two
   engines and *asserts the compiled instruction sets* via `ggml_cpu_has_*`:
   - `tarjama-engine-safe`  - plain SSE2, runs on **every** x86_64 CPU
   - `tarjama-engine-fast`  - AVX2+FMA+F16C, auto-selected on modern CPUs
2. **Crash isolation** - the engine runs as a **separate process**; the GUI
   streams JSON events from it. A native crash can never close the app:
   the UI shows a readable error and automatically retries with the safe engine.
3. **Verified behavior** - CI actually *runs* the shipped exes on the Windows
   runner: engine selftest (ASR + MT), full pipeline on a generated test video
   for both engines, and a scripted GUI autotest (Linux, under Xvfb) that
   also simulates a fast-engine crash to prove the automatic fallback.

## Layout

```
crates/tarjama-core     shared types, JSON event protocol, SRT/glossary/wav
crates/tarjama-engine   whisper.cpp (ASR) + ONNX OPUS-MT tr->ar (translation)
crates/tarjama-gui      iced native UI (tarjama.exe), spawns the engine
vendor/whisper-rs-sys   patched fork (TARJAMA_GGML_* pass-through)
```

## Models (bundled, compressed)

Only three whisper models are included, quantized q5_1:
Tiny (32 MB), Base (60 MB), Small (190 MB) - plus OPUS-MT tr->ar int8 ONNX.
Medium and Large-v3 Turbo are intentionally NOT part of this app.

## CI

`.github/workflows/build.yml` builds everything on GitHub, caches models and
ffmpeg, publishes a portable zip as artifact + GitHub release.
