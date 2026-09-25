Tarjama Studio Native v1.2 - portable, fully offline
=====================================================

WHAT'S NEW IN v1.2 - GPU ACCELERATION
  - The app now transcribes on your GRAPHICS CARD via the Vulkan engine
    (tarjama-engine-gpu.exe). This works with Intel, AMD and NVIDIA GPUs,
    including integrated laptop graphics. A 2-minute video that used to take
    minutes on the CPU engine should now finish in a fraction of the time.
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
  Tiny  (32 MB)  - fastest, lowest quality
  Base  (60 MB)  - balanced
  Small (190 MB) - best quality (recommended for TV series)

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
