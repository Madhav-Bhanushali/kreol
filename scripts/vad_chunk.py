"""Silence/VAD-based audio chunking (see project doc "Chunking approach").

Rules implemented:
- Prefer silence-based splitting: speech regions separated by pauses (RMS-based
  VAD, ~librosa.effects.split equivalent) are merged into chunks.
- A chunk never exceeds MAX_CHUNK_SECONDS (28s default; headroom under the 30s
  Gemma encoder limit).
- If a single speech interval would exceed the limit, it is hard-split at the
  nearest low-energy point (windowed RMS minimum around the target boundary),
  not mid-word if avoidable.

Output: list of (start_sec, end_sec) chunks.
"""
import argparse
from pathlib import Path

import numpy as np


def load_wav(path, sr=16000):
    """Load a wav file to mono float32 at target sample rate."""
    import soundfile as sf

    y, orig_sr = sf.read(path, dtype="float32", always_2d=True)
    if y.shape[1] > 1:
        y = y.mean(axis=1)
    else:
        y = y.reshape(-1)
    if orig_sr != sr:
        try:
            import librosa

            y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
        except Exception:
            n = int(len(y) * sr / orig_sr)
            xp = np.arange(len(y))
            y = np.interp(np.linspace(0, len(y) - 1, n), xp, y).astype(np.float32)
    return np.ascontiguousarray(y), sr


def _frame_rms_db(y, frame, hop):
    n = len(y)
    if n < frame:
        frame = max(n, 1)
    frames = np.lib.stride_tricks.sliding_window_view(y, frame)[::hop]
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-10)
    return 20.0 * np.log10(rms + 1e-10)


def _nearest_low_energy_split(y, sr, start, end, target_sec, top_db, window_sec=0.75):
    """Split [start,end] near target_sec using the lowest-RMS point in a window."""
    target = start + target_sec * sr
    lo = max(start, int(target - window_sec * sr))
    hi = min(end, int(target + window_sec * sr))
    if hi - lo < sr // 10:
        return lo
    seg = y[lo:hi]
    frame, hop = int(0.02 * sr), int(0.01 * sr)
    db = _frame_rms_db(seg, frame, hop)
    best = int(np.argmin(db)) * hop
    return lo + best


def silence_chunks(y, sr, max_seconds=28.0, top_db=35.0, min_silence_sec=0.25):
    """Split a waveform into speech chunks under max_seconds using an RMS VAD."""
    n = len(y)
    if n == 0:
        return []
    frame = int(0.02 * sr)          # 20 ms
    hop = int(0.01 * sr)            # 10 ms
    db = _frame_rms_db(y, frame, hop)
    thr = db.max() - top_db
    speech = db > thr

    # speech intervals in seconds (collapse gaps shorter than min_silence_sec)
    intervals = []
    start = None
    for i, active in enumerate(speech):
        t0, t1 = i * hop / sr, (i + 1) * hop / sr
        if active and start is None:
            start = t0
        elif not active and start is not None:
            intervals.append((start, t0))
            start = None
    if start is not None:
        intervals.append((start, n / sr))

    merged = []
    for s, e in intervals:
        if merged and s - merged[-1][1] <= min_silence_sec:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    # enforce the max-30s headroom: hard-split over-long intervals at nearest
    # low-energy point, then merge neighbours into <= max_seconds chunks
    splits = []
    for s, e in merged:
        while e - s > max_seconds:
            b = _nearest_low_energy_split(y, sr, int(s * sr), int(e * sr),
                                          max_seconds, top_db)
            if b <= int(s * sr) + sr // 10:
                b = int(s * sr) + int(max_seconds * sr)
            splits.append((s, b / sr))
            s = b / sr
        splits.append((s, e))

    chunks = []
    for s, e in splits:
        if chunks and e - chunks[-1][0] <= max_seconds:
            chunks[-1] = (chunks[-1][0], e)
        else:
            chunks.append((s, e))
    return [(round(s, 3), round(e, 3)) for s, e in chunks if e - s > 0.05]


def chunk_file(path, max_seconds=28.0, top_db=35.0):
    y, sr = load_wav(path)
    return silence_chunks(y, sr, max_seconds=max_seconds, top_db=top_db), sr


def main():
    ap = argparse.ArgumentParser(description="Split long audio into silence-based chunks under ~28s")
    ap.add_argument("audio")
    ap.add_argument("--max-seconds", type=float, default=28.0)
    ap.add_argument("--top-db", type=float, default=35.0)
    args = ap.parse_args()

    chunks, sr = chunk_file(args.audio, max_seconds=args.max_seconds, top_db=args.top_db)
    total = float(Path(args.audio).stat().st_size)
    print(f"sr={sr} chunks={len(chunks)}")
    for i, (s, e) in enumerate(chunks, 1):
        print(f"  [{i}] {s:.2f}-{e:.2f}  ({e - s:.2f}s)")


if __name__ == "__main__":
    main()