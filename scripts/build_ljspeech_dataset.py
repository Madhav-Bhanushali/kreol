"""Build an LJSpeech-format dataset from WorldSpeech `mfe_mu` for Chatterbox
fine-tuning (md#2 + md#3).

- Loads the HF dataset with the audio column kept as raw bytes (bypasses the
  `torchcodec` audio decoder, which is broken on this box), decodes with
  soundfile.
- Filters by CER (default < 0.3, matching the WorldSpeech paper's quality
  convention).
- Writes `wavs/<id>.wav` (24 kHz mono PCM-16) + `metadata.csv` with rows
  `id|human_transcript|human_transcript`.

Run on the box:
    HF_DATASETS_DISABLE_TORCHCODEC=1 python scripts/build_ljspeech_dataset.py \
        --out data/worldspeech_mfe_ljspeech
"""
import argparse
import io
import re
import time
from pathlib import Path

import datasets
import numpy as np
import soundfile as sf
from datasets import Features, Value, load_dataset

SAMPLING_RATE = 24000
_CLEAN_RE = re.compile(r"\s+")


def clean(text):
    return _CLEAN_RE.sub(" ", (text or "").replace("|", " ")).strip()


def build(out_dir, cer_threshold, max_rows=None):
    info = datasets.get_dataset_config_info("disco-eth/WorldSpeech", "mfe_mu")
    feats = dict(info.features)
    feats["audio"] = {"bytes": Value("binary"), "path": Value("string")}
    ds = load_dataset("disco-eth/WorldSpeech", "mfe_mu", split="train",
                      features=Features(feats))

    keep = (np.array(ds["cer"], dtype=float) < cer_threshold)
    if max_rows:
        idx = np.where(keep)[0][:max_rows]
    else:
        idx = np.where(keep)[0]

    wavs = Path(out_dir) / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    meta_path = Path(out_dir) / "metadata.csv"

    t0 = time.time()
    n_skipped = 0
    total_secs = 0.0
    rows = []
    with meta_path.open("w", encoding="utf-8") as mf:
        for i, di in enumerate(idx, 1):
            row = ds[int(di)]
            raw = row["audio"]["bytes"]
            if raw is None:
                n_skipped += 1
                continue
            arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
            if sr != SAMPLING_RATE:
                # native data is 24 kHz; resample defensively if ever not
                from scipy.signal import resample_poly
                arr = resample_poly(arr, SAMPLING_RATE, sr)
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            wav_id = f"mfe_mu_{i:06d}"
            wav_path = wavs / f"{wav_id}.wav"
            sf.write(wav_path, arr, SAMPLING_RATE, subtype="PCM_16")
            text = clean(row["human_transcript"])
            if not text:
                n_skipped += 1
                wav_path.unlink(missing_ok=True)
                continue
            mf.write(f"{wav_id}|{text}|{text}\n")
            total_secs += len(arr) / SAMPLING_RATE
            rows.append((wav_id, text))
            if i % 500 == 0:
                print(f"{i}/{len(idx)} rows in {time.time()-t0:.0f}s", flush=True)

    print(f"\nbuilt {len(rows)} rows -> {out_dir}  ({total_secs/3600:.1f} hours)  "
          f"skipped={n_skipped}  in {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser(description="Build LJSpeech dataset from WorldSpeech mfe_mu")
    ap.add_argument("--out", default="data/worldspeech_mfe_ljspeech")
    ap.add_argument("--cer", type=float, default=0.3, help="CER threshold to keep rows")
    ap.add_argument("--max-rows", type=int, default=None, help="cap for a dry run")
    args = ap.parse_args()
    build(args.out, args.cer, args.max_rows)


if __name__ == "__main__":
    main()