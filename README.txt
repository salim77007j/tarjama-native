Tarjama Studio Native v1.1 - portable, fully offline
=====================================================

WHAT'S NEW IN v1.1
  - Fixed: crash on translate (illegal-instruction) on CPUs without AVX2/AVX512.
    The program now ships TWO engines and picks the right one automatically:
      tarjama-engine-fast.exe  AVX2 build - used automatically on modern CPUs
      tarjama-engine-safe.exe  plain SSE2 build - runs on EVERY x86_64 CPU
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

  No internet is needed, ever. Everything runs on your CPU.

MODELS (bundled, compressed - nothing else exists in this app)
  Tiny  (32 MB)  - fastest, lowest quality
  Base  (60 MB)  - balanced
  Small (190 MB) - best quality (recommended for TV series)

COMMAND LINE
  tarjama.exe --cli --input video.mp4 --model small --out C:\subs\ep1
  tarjama.exe --selftest    (checks all models + full pipeline)

TROUBLESHOOTING
  - "models folder not found": keep tarjama.exe in the same folder as models/.
  - If the engine ever crashes, the app retries with the safe engine
    automatically; the log pane shows exactly what happened.
  - Long files: Tiny is 5-10x realtime; Base/Small are 2-5x realtime on a
    modern 4-core CPU (a bit slower on the compatibility engine).

Included third-party software: whisper.cpp (MIT), whisper ggml models (MIT),
OPUS-MT tr-ar (CC-BY), ONNX Runtime (MIT), Noto fonts (OFL), ffmpeg (GPL, used
as an external program).
