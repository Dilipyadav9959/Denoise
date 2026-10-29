"""Batch noise suppression with DeepFilterNet (CPU).

Usage:
    python denoise.py                      # processes everything in the input folder
    python denoise.py a.wav b.mp3          # specific files
    python denoise.py -o out --pf          # custom output dir + postfilter

Any format ffmpeg understands works (mp3, mp4 reels, m4a, aac, opus, ...).
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from df.enhance import enhance, init_df, load_audio, save_audio

EXTS = {".wav", ".flac", ".ogg", ".mp3", ".mp4", ".m4a", ".aac", ".opus", ".mkv", ".webm"}


def ffmpeg_to_wav(src, sr):
    """Decode any media file to a temp 48 kHz mono wav using bundled ffmpeg."""
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = [ff, "-y", "-loglevel", "error", "-i", str(src),
           "-vn", "-ac", "1", "-ar", str(sr), tmp.name]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        os.unlink(tmp.name)
        raise RuntimeError(f"ffmpeg could not read {src}:\n{r.stderr}")
    return tmp.name


def load_any(path, sr):
    """Plain .wav goes straight in; everything else (or failures) via ffmpeg."""
    if path.suffix.lower() == ".wav":
        try:
            audio, _ = load_audio(str(path), sr)
            return audio.mean(dim=0, keepdim=True)
        except Exception:
            pass
    tmp = ffmpeg_to_wav(path, sr)
    try:
        audio, _ = load_audio(tmp, sr)
    finally:
        os.unlink(tmp)
    return audio.mean(dim=0, keepdim=True)


def collect(inputs):
    files = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in EXTS)
        elif p.is_file():
            files.append(p)
        else:
            print(f"[warn] not found: {p}")
    return files


def main():
    ap = argparse.ArgumentParser(description="DeepFilterNet batch denoiser")
    ap.add_argument("inputs", nargs="*", default=["input"], help="files or folders")
    ap.add_argument("-o", "--out-dir", default="output")
    ap.add_argument("--pf", action="store_true",
                    help="postfilter: removes more noise, may over-suppress")
    ap.add_argument("--atten", type=float, default=None,
                    help="max noise attenuation in dB (e.g. 20) for a more natural sound")
    a = ap.parse_args()

    files = collect(a.inputs)
    if not files:
        print("No audio files found. Put audio files in the 'input' folder.")
        sys.exit(1)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model (first run downloads weights, needs internet)...")
    model, df_state, _ = init_df(post_filter=a.pf)
    sr = df_state.sr()  # 48000

    for f in files:
        print(f"Processing {f.name} ...", end=" ", flush=True)
        try:
            audio = load_any(f, sr)
        except Exception as e:
            print(f"SKIPPED\n  {e}")
            continue
        t0 = time.time()
        enhanced = enhance(model, df_state, audio, atten_lim_db=a.atten)
        out = out_dir / f"{f.stem}_clean.wav"
        save_audio(str(out), enhanced, sr)
        dur = audio.shape[-1] / sr
        print(f"done ({dur:.1f}s audio in {time.time() - t0:.1f}s) -> {out}")


if __name__ == "__main__":
    main()