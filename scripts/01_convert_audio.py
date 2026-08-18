import argparse
import subprocess
from pathlib import Path


def convert(src: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (src.stem + ".wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "24000", str(out)],
        check=True,
        capture_output=True,
    )
    return out


def main():
    ap = argparse.ArgumentParser(description="Convert MP3 chapter audio to 24 kHz mono WAV (requires ffmpeg).")
    ap.add_argument("audio_dir", type=Path, help="Folder with the chapter .mp3 files (e.g. MFEBSMN2DA/.../*.mp3)")
    ap.add_argument("--out_dir", type=Path, default=Path("data/chapters_24k"))
    args = ap.parse_args()

    files = sorted(args.audio_dir.glob("*.mp3"))
    if not files:
        raise SystemExit(f"no .mp3 files found in {args.audio_dir}")
    for f in files:
        convert(f, args.out_dir)
        print("converted", f.name)
    print(f"done: {len(files)} files in {args.out_dir}")


if __name__ == "__main__":
    main()