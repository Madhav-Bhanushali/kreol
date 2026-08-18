import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        description="Drop low-confidence / implausible verse segments from alignment results."
    )
    ap.add_argument("align_dir", type=Path, help="data/alignments")
    ap.add_argument("--out_dir", type=Path, default=Path("data/alignments_filtered"))
    ap.add_argument("--min_score", type=float, default=0.60, help="drop verses with mean word score below this")
    ap.add_argument("--min_dur", type=float, default=1.0)
    ap.add_argument("--max_dur", type=float, default=15.0)
    ap.add_argument("--min_chars_per_sec", type=float, default=3.0, help="sanity: speaking-rate lower bound")
    ap.add_argument("--max_chars_per_sec", type=float, default=22.0, help="sanity: speaking-rate upper bound")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    kept = dropped = 0

    for aj in sorted(args.align_dir.glob("*.json")):
        with open(aj, encoding="utf-8") as f:
            verses = json.load(f)
        out = {}
        for num, seg in verses.items():
            if seg["status"] != "ok":
                dropped += 1
                continue
            dur = seg["end"] - seg["start"]
            chars = len(seg["text"])
            if seg["score"] < args.min_score or not (args.min_dur <= dur <= args.max_dur):
                dropped += 1
                continue
            cps = chars / dur
            if not (args.min_chars_per_sec <= cps <= args.max_chars_per_sec):
                dropped += 1
                continue
            out[num] = seg
            kept += 1
        with open(args.out_dir / aj.name, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"kept {kept}, dropped {dropped}")


if __name__ == "__main__":
    main()