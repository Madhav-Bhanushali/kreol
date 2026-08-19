"""To-do #1: test Gemma's ASR on sample Morisyen audio.

Runs the Gemma audio endpoint on a real Morisyen clip and prints the transcript
so we can judge whether Gemma can transcribe Morisyen at all. Pass --asr mms to
compare against the mms-1b-all fallback on the same file.
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Test Gemma ASR on sample Morisyen audio")
    ap.add_argument("audio", help="sample Morisyen audio (wav/mp3)")
    ap.add_argument("--asr", choices=["gemma", "mms"], default="gemma",
                    help="'gemma' = Gemma audio endpoint; 'mms' = mms-1b-all fallback comparison")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-seconds", type=float, default=28.0)
    ap.add_argument("--top-db", type=float, default=35.0)
    args = ap.parse_args()

    from gemma_asr import transcribe_file

    transcribe_file(args.audio, asr=args.asr, max_seconds=args.max_seconds,
                    top_db=args.top_db, out=args.out)


if __name__ == "__main__":
    main()