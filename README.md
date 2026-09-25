# Tarjama Studio Native

Standalone, fully offline desktop app: **Turkish video/audio -> Arabic subtitles**.
Pure native Rust (no web technologies, no browser, no installer, no internet).

- Whisper speech recognition (whisper.cpp via whisper-rs) - models **Tiny / Base / Small** (quantized q5_1) bundled. No Medium, no Large.
- Turkish -> Arabic translation (OPUS-MT, int8-quantized ONNX Runtime) bundled.
- ffmpeg.exe (GPL build) bundled - reads every common video/audio format.
- Native GUI (iced) - renders Arabic (RTL) and Turkish correctly.
- Runs entirely on CPU, tuned to work on old/low-end machines (SSE-only build).

## Use (Windows)

1. Download `TarjamaNative-Windows-x64.zip` from Releases and unzip it anywhere.
2. Double-click `tarjama.exe`.
3. Pick a video/audio file, choose a model, press **TRANSLATE**.
4. Export SRT (Arabic), bilingual SRT, VTT or TXT - files are written next to your video.

If SmartScreen warns about an unsigned app: click "More info" -> "Run anyway".

## Build from source

Requires Rust (stable) and cmake. `cargo build --release` - GitHub Actions does this automatically on every push (see `.github/workflows/build.yml`), runs a full pipeline selftest on Windows and Linux runners, and publishes a portable zip.

## CLI

```
tarjama.exe --cli --input movie.mp4 --model small --out C:\subs\ep1
tarjama.exe --selftest
```

## Model licenses

- Whisper models: MIT (OpenAI) via whisper.cpp ggml quantizations
- OPUS-MT (Helsinki-NLP/opus-mt-tr-ar): CC-BY 4.0 / Apache-2.0 (OPUS data)
- Noto fonts: OFL 1.1
- ffmpeg.exe: GPL v3 build (BtbN) - kept as a separate process, not linked
