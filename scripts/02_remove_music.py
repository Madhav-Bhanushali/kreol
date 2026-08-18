import argparse
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Optional: remove background music/SFX from the Drama recording with Demucs "
            "(keep only the vocals stem). Run this BEFORE alignment if music is present."
        )
    )
    ap.add_argument("wav_dir", type=Path, help="data/chapters_24k")
    ap.add_argument("--out_dir", type=Path, default=Path("data/chapters_clean"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    import soundfile as sf
    import torchaudio

    model = get_model("htdemucs").to(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for wav in sorted(args.wav_dir.glob("*.wav")):
        x, sr = torchaudio.load(wav)
        if sr != 44100:
            x = torchaudio.functional.resample(x, sr, 44100)
        with torch.no_grad():
            stems = apply_model(model, x.unsqueeze(0).to(args.device), split=True, progress=True)[0]
        vocals = stems[3].cpu()  # htdemucs stem order: drums, bass, other, vocals
        torchaudio.save(args.out_dir / wav.name, vocals, 44100)
        print("cleaned", wav.name)


if __name__ == "__main__":
    main()