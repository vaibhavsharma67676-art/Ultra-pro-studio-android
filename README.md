---
title: Aurashiftai
sdk: docker
emoji: ⚡
colorTo: blue
short_description: Next-Gen Creative & Media Suite
colorFrom: indigo
---
# AuraShift AI — Ultra Pro Studio upgrade

Android Capacitor wrapper and phone-based cloud APK build instructions: see [`README_ANDROID.md`](README_ANDROID.md).

This package upgrades **only the Ultra Pro Studio** workflow in the supplied AuraShift AI project. The other utility-tool implementations (image, PDF, QR, background-removal, and watermark tools) are retained unchanged.

## Included files

- `indexx.html` — updated Ultra Pro Studio controls and workflow.
- `server.py` — Studio audio DSP, optional BPM analysis endpoint, multi-track mix and export changes.
- `requirements.txt` — copied unchanged from the supplied project.
- `Dockerfile` — copied unchanged in content from the supplied `Dockerfile.txt`, renamed to the conventional Docker build filename.

## Studio capabilities added

- Global pitch transposition in semitones, plus automatic pitch correction constrained by selected key/scale.
- Optional noise reduction and adjustable vocal time-stretch.
- Manual compressor threshold, ratio, attack and release controls.
- Optional backing-vocal/double layer with level, pan and timing offset.
- Instrumental timing offset and vocal-triggered instrumental ducking.
- Instrumental BPM estimation button for setting the tempo-synced delay clock.
- Browser-local save/load of Studio chain settings (audio files are not saved).
- 16-bit WAV, 24-bit WAV, MP3 320 kbps and ZIP export containing the master plus available pre-master stems.
- Output sample peak, RMS and loudness measurements in the render preview.

## Run

Put these files in the same project directory and use the supplied Dockerfile:

```bash
docker build -t aurashift-ai-studio .
docker run --rm -p 7860:7860 aurashift-ai-studio
```

Open `http://localhost:7860`.

## Audio notes and limitations

- Start the lead vocal, instrumental and backing-vocal files at the same musical position where possible; use track offset controls for small alignment corrections.
- The key/scale controls influence automatic note correction. Global pitch transposition changes the lead vocal only, so check it against the instrumental before finalizing.
- High noise-reduction settings, large pitch shifts and large time-stretches can create artifacts. Always use the A/B preview.
- The ZIP's vocal/backing/instrumental stems are pre-master stems. The mastered mix has bus compression and limiting, so summing the stems will not exactly recreate the master.
- The displayed peak ceiling is a sample-peak safeguard; this build does **not** claim to provide a dedicated oversampled true-peak compliance meter.
- This is an enhanced browser-based vocal-production and three-layer mix workflow, not a complete commercial DAW: graphical piano-roll note-by-note editing, multi-take comping and arbitrary drawn automation lanes are not implemented in this pass.