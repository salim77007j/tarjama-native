Tarjama Studio Native v1.3 - portable, fully offline
=====================================================

WHAT'S NEW IN v1.3 - SUBTITLE QUALITY OVERHAUL
  - Whisper models upgraded from 5-bit (q5_1) to 8-bit (q8_0) quantization:
    near-lossless, the same precision class as the web studio's int8 models.
  - Whisper now decodes with BEAM SEARCH (beam 5) instead of greedy, and no
    longer carries previous text into the next window (prevents hallucination
    loops on music, silence and repeats) - matching the web pipeline.
  - Translation engine rebuilt: OPUS-MT (ONNX int8) now decodes with BEAM
    SEARCH (beam 4, length penalty 0.2) instead of greedy. This was the
    biggest single quality jump, verified sentence-by-sentence against
    reference translations.
  - Stray quotation-mark artifacts from the ASR are stripped before
    translation and before writing subtitles.
  - TIP: for the best subtitle quality pick the SMALL model.

WHAT WAS NEW IN v1.2 - GPU ACCELERATION
  - The app transcribes on your GRAPHICS CARD via the Vulkan engine
    (tarjama-engine-gpu.exe). Works with Intel, AMD and NVIDIA GPUs,
    including integrated laptop graphics.
  - The app picks the best engine automatically for your machine:
      tarjama-engine-gpu.exe  GPU (Vulkan) - used when a supported GPU exists
      tarjama-engine-fast.exe AVX2 CPU build - used on modern CPUs without GPU
      tarjama-engine-safe.exe plain SSE2 CPU build - runs on EVERY x86_64 CPU
  - The log pane shows which backend really ran:
      "compute backend: GPU (Vulkan) - <GPU name>"  or  "compute backend: CPU"
  - If the GPU engine cannot start or ever crashes, the app retries
    automatically with the next engine (fast, then safe). Nothing breaks.
  - Advanced: set the environment variable TARJAMA_NO_GPU=1 to force CPU.

WHAT WAS NEW IN v1.1
  - Fixed: crash on translate (illegal-instruction) on CPUs without AVX2/AVX512.
  - The engine runs as a separate process: if it ever fails, the window shows a
    readable error and automatically retries with the safe engine. The app no
    longer closes when something goes wrong.
  - Cancel button really stops processing now. Progress shows live elapsed time.

RUN
  1. Unzip anywhere (keep tarjama.exe together with the models folder).
  2. Double-click tarjama.exe
     (If SmartScreen warns: "More info" -> "Run anyway" - the app is unsigned.)
  3. Choose a Turkish video/audio file, pick a model, press TRANSLATE.
  4. Export buttons write subtitle files next to your video:
       <name>.ar.srt          Arabic subtitles
       <name>.bilingual.srt   Arabic + Turkish
       <name>.ar.vtt          WebVTT
       <name>.ar.txt          plain Arabic text
       <name>.tr.txt          Turkish transcript

  No internet is needed, ever.

MODELS (bundled, compressed - nothing else exists in this app)
  Tiny  (43 MB)  - fastest, lowest quality
  Base  (80 MB)  - balanced
  Small (270 MB) - best quality (recommended for TV series)

COMMAND LINE
  tarjama.exe --cli --input video.mp4 --model small --out C:\subs\ep1
  tarjama.exe --selftest    (checks all models + full pipeline)

TROUBLESHOOTING
  - "models folder not found": keep tarjama.exe in the same folder as models/.
  - Speed tip: for best GPU speed, update your graphics driver from your
    laptop/PC maker (or intel.com / amd.com / nvidia.com). Older drivers can
    be slower or lack Vulkan; the app then falls back to the CPU engine.
  - If the engine ever crashes, the app retries with the next engine
    automatically; the log pane shows exactly what happened.
  - Long files: Tiny is 5-10x realtime; Base/Small are 2-5x realtime on a
    modern 4-core CPU (a bit slower on the compatibility engine).

Included third-party software: whisper.cpp (MIT), whisper ggml models (MIT),
OPUS-MT tr-ar (CC-BY), ONNX Runtime (MIT), Noto fonts (OFL), ffmpeg (GPL, used
as an external program).
