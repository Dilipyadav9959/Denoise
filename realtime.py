"""EXPERIMENTAL live mic denoiser for Windows.

mic -> DeepFilterNet -> virtual cable (VB-Cable) -> Zoom/Discord/OBS

The Python API is not a streaming API, so this re-processes a rolling window
and only emits the part that has enough "future" context. Expect roughly
0.3-0.5 s latency (fine for recording/streaming, noticeable in calls).

    python realtime.py --list
    python realtime.py --in-dev 1 --out-dev 7
"""
import argparse
import queue
import threading
import time

import numpy as np
import sounddevice as sd
import torch

from df.enhance import enhance, init_df


class Fifo:
    """Tiny thread-safe float32 FIFO for the output callback."""

    def __init__(self):
        self.buf = np.zeros(0, dtype=np.float32)
        self.lock = threading.Lock()

    def push(self, x):
        with self.lock:
            self.buf = np.concatenate([self.buf, x])

    def pull(self, n):
        with self.lock:
            out = self.buf[:n]
            self.buf = self.buf[n:]
        if len(out) < n:  # underrun -> pad with silence
            out = np.concatenate([out, np.zeros(n - len(out), dtype=np.float32)])
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list audio devices and exit")
    ap.add_argument("--in-dev", type=int, help="microphone device index")
    ap.add_argument("--out-dev", type=int, help="output device index (e.g. CABLE Input)")
    ap.add_argument("--hop", type=float, default=0.16, help="seconds per processing step")
    ap.add_argument("--look", type=float, default=0.10, help="lookahead seconds (quality vs latency)")
    ap.add_argument("--window", type=float, default=1.5, help="context window seconds")
    ap.add_argument("--atten", type=float, default=None, help="max attenuation dB")
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()

    if a.list:
        print(sd.query_devices())
        return
    if a.in_dev is None or a.out_dev is None:
        print("Specify --in-dev and --out-dev (see --list).")
        return

    torch.set_num_threads(a.threads)
    model, df_state, _ = init_df()
    sr = df_state.sr()
    hop_n, look_n, win_n = int(a.hop * sr), int(a.look * sr), int(a.window * sr)
    assert hop_n + look_n <= win_n, "window must be larger than hop + look"

    in_q = queue.Queue()
    out_fifo = Fifo()
    out_fifo.push(np.zeros(hop_n, dtype=np.float32))  # jitter buffer
    stop = threading.Event()

    def in_cb(indata, frames, t, status):
        in_q.put(indata[:, 0].copy())

    def out_cb(outdata, frames, t, status):
        outdata[:, 0] = out_fifo.pull(frames)

    def worker():
        buf = np.zeros(win_n, dtype=np.float32)
        pending = 0
        while not stop.is_set():
            try:
                chunk = in_q.get(timeout=0.1)
            except queue.Empty:
                continue
            buf = np.concatenate([buf, chunk])[-win_n:]
            pending += len(chunk)
            if pending < hop_n:
                continue
            n = min(pending, win_n - look_n)
            t0 = time.time()
            with torch.inference_mode():
                enh = enhance(model, df_state, torch.from_numpy(buf).unsqueeze(0),
                              atten_lim_db=a.atten)
            enh = enh.squeeze(0).numpy()
            seg = np.clip(enh[len(enh) - look_n - n: len(enh) - look_n], -1.0, 1.0)
            out_fifo.push(seg.astype(np.float32))
            pending = 0
            load = (time.time() - t0) / (n / sr)
            if load > 0.9:
                print(f"[warn] CPU too slow (load {load:.2f}x). Increase --hop or lower --window.")

    threading.Thread(target=worker, daemon=True).start()
    block = int(0.02 * sr)
    with sd.InputStream(device=a.in_dev, samplerate=sr, channels=1, dtype="float32",
                        blocksize=block, callback=in_cb), \
         sd.OutputStream(device=a.out_dev, samplerate=sr, channels=1, dtype="float32",
                         blocksize=block, callback=out_cb):
        print("Running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
    stop.set()


if __name__ == "__main__":
    main()
