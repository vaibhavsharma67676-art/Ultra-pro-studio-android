AuraShift AI — Ultra Pro Studio update

Files in this bundle:
- indexx.html: existing website with the three new browser-native audio plugins
- server.py: original Flask app preserved, with Split Stems and Auto Mix processing removed
- Dockerfile: original deployment configuration, unchanged
- requirements.txt: original Python dependencies, unchanged

Deployment:
1. Replace the same-named files in the existing Hugging Face Space repository with these files.
2. Keep the Dockerfile and requirements.txt at repository root.
3. Rebuild/restart the Space.

Ultra Pro Studio now keeps Manual Vocal Mod and Studio Clean, and adds:
- Retro Tape & Vinyl Saturator — effect render/export as WAV
- Lo-Fi Ambient Space Delay — stereo echo/feedback render/export as WAV
- Dual-Oscillator Analog Style Synth — playable two-octave instrument with performance recording/export as WAV

Tape, Delay and Synth processing runs in the visitor's browser using Web Audio API; those audio files are not uploaded to the Flask backend. Browser audio support is required. Audio-effect source files are limited to 60 MB and 5 minutes to reduce mobile memory failures. Synth performance recording automatically stops at 2 minutes.
