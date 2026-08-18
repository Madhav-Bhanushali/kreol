import argparse
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        description="Run F5-TTS prepare_csv_wavs.py on metadata.csv to build raw.arrow + duration.json + vocab.txt."
    )
    ap.add_argument("--metadata", type=Path, default=Path("data/f5tts/metadata.csv"))
    ap.add_argument("--f5tts_dir", type=Path, required=True, help="Path to your F5-TTS checkout (editable install)")
    ap.add_argument("--dataset_name", default="MFEBSM")
    ap.add_argument("--tokenizer", default="pinyin")
    args = ap.parse_args()

    out_dir = args.f5tts_dir / "data" / f"{args.dataset_name}_{args.tokenizer}"
    cmd = [
        sys.executable,
        "src/f5_tts/train/datasets/prepare_csv_wavs.py",
        str(args.metadata.resolve()),
        str(out_dir),
    ]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, cwd=args.f5tts_dir, check=True)
    print(f"prepared dataset in: {out_dir}")
    print("finetune with: --dataset_name", args.dataset_name, "--tokenizer", args.tokenizer)


if __name__ == "__main__":
    main()