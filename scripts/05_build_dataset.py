import argparse
import json
import re
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def clean_text(s: str) -> str:
    s = re.sub(r"^\s*\d+\s*", "", s)  # leading verse number
    s = re.sub(r"[\[\]()]", " ", s)  # footnote / cross-reference markers
    s = re.sub(r"\s+", " ", s).strip()
    return s


def write_csv(path: Path, rows: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("audio_file|text\n")
        f.write("\n".join(rows))
        f.write("\n")


def main():
    ap = argparse.ArgumentParser(
        description="Slice verse segments into 24 kHz wavs and write metadata.csv (audio_file|text)."
    )
    ap.add_argument("align_dir", type=Path, help="data/alignments_filtered")
    ap.add_argument("wav_dir", type=Path, help="data/chapters_clean (must match wav stems used in alignment)")
    ap.add_argument("--out_dir", type=Path, default=Path("data/f5tts"))
    ap.add_argument("--val_fraction", type=float, default=0.08, help="hold-out fraction for evaluation")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_wavs = args.out_dir / "wavs"
    out_wavs.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    train_rows, val_rows = [], []

    for aj in sorted(args.align_dir.glob("*.json")):
        stem = aj.stem
        wav_path = args.wav_dir / f"{stem}.wav"
        if not wav_path.exists():
            print(f"SKIP (no wav): {stem}")
            continue
        audio, sr = librosa.load(wav_path, sr=24000, mono=True)
        with open(aj, encoding="utf-8") as f:
            verses = json.load(f)
        for num in sorted(verses, key=lambda k: int(re.sub(r"\D", "", k) or 0)):
            seg = verses[num]
            text = clean_text(seg["text"])
            if not text:
                continue
            s, e = int(seg["start"] * sr), int(seg["end"] * sr)
            if not (s < e <= len(audio)):
                continue
            clip, _ = librosa.effects.trim(audio[s:e], top_db=30)  # trim leading/trailing silence
            if clip.size / sr < 1.0 or clip.size / sr > 15.0:
                continue
            clip = clip / max(np.abs(clip).max(), 1e-8)  # peak-normalize (approx -3 dBFS)
            name = f"{stem}_{int(re.sub(r'\\D', '', num) or 0):03d}.wav"
            out = out_wavs / name
            sf.write(out, clip, sr)
            row = f"{out.resolve()}|{text}"
            (val_rows if rng.random() < args.val_fraction else train_rows).append(row)

    write_csv(args.out_dir / "metadata.csv", train_rows)
    write_csv(args.out_dir / "metadata_val.csv", val_rows)
    print(f"train: {len(train_rows)}, val: {len(val_rows)} -> {args.out_dir}")


if __name__ == "__main__":
    main()