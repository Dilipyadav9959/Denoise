# my-denoiser (DeepFilterNet on Windows, CPU only, free)

## 1. Install (once)
Double-click `setup.bat`. If it says Python 3.11 is missing, run in a terminal:
    winget install -e --id Python.Python.3.11
then close the terminal and run setup.bat again. (Python 3.12+ will NOT work with the pinned torch.)

## 2. Clean audio files
Put .wav/.mp3/.flac files in `input\`, double-click `denoise.bat`.
Results appear in `output\` as `<name>_clean.wav`.
Options: `denoise.bat --pf` (stronger), `denoise.bat --atten 20` (more natural).

## 3. Live mic (experimental)
1. Install VB-Cable (free): https://vb-audio.com/Cable/  then reboot.
2. `realtime.bat --list` and note your mic index and the "CABLE Input" output index.
   Pick entries from the same host API (e.g. all "MME").
3. `realtime.bat --in-dev <mic> --out-dev <cable input>`
4. In Zoom/Discord/OBS choose "CABLE Output" as the microphone.
If you hear glitches: raise `--hop 0.25`. If quality is off at word ends: raise `--look 0.15`.
