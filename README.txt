Tarjama Studio Native v1.0 - portable, fully offline
=====================================================

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

MODELS
  Tiny  (32 MB)  - fastest, lowest quality
  Base  (60 MB)  - balanced
  Small (190 MB) - best quality (recommended for TV series)

COMMAND LINE
  tarjama.exe --cli --input video.mp4 --model small --out C:\subs\ep1
  tarjama.exe --selftest    (checks all models + full pipeline)

TROUBLESHOOTING
  - "models folder not found": keep tarjama.exe in the same folder as models/.
  - If the window is blank on very old GPUs, the app automatically falls back
    to software rendering.
  - Long files: processing speed is roughly 2-5x realtime for Base/Small on a
    modern 4-core CPU; Tiny is 5-10x realtime.

Included third-party software: whisper.cpp (MIT), whisper ggml models (MIT),
OPUS-MT tr-ar (CC-BY), ONNX Runtime (MIT), Noto fonts (OFL), ffmpeg (GPL, used
as an external program).
