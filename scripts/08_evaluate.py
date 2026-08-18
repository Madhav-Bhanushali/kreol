import argparse
import json
import re
from pathlib import Path


def normalize(s: str) -> str:
    return re.sub(r"[^\w]", "", s.lower())


def cer(ref: str, hyp: str) -> float:
    import jiwer

    return jiwer.cer(ref, hyp)


def wer(ref: str, hyp: str) -> float:
    import jiwer

    return jiwer.wer(ref, hyp)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate a fine-tuned F5-TTS checkpoint: synthesize held-out verses, "
            "transcribe with facebook/mms-1b-all, report CER/WER."
        )
    )
    ap.add_argument("--f5tts_dir", type=Path, required=True, help="Path to F5-TTS checkout")
    ap.add_argument("--ckpt_dir", type=Path, required=True, help="e.g. ckpts/MFEBSM/F5TTS_v1_Base")
    ap.add_argument("--dataset_name", default="MFEBSM")
    ap.add_argument("--metadata_val", type=Path, default=Path("data/f5tts/metadata_val.csv"))
    ap.add_argument("--ref_wav", type=Path, help="clean narrator reference wav for voice cloning")
    ap.add_argument("--ref_text", type=str, help="transcript of ref_wav")
    ap.add_argument("--num_samples", type=int, default=20)
    args = ap.parse_args()

    import sys

    sys.path.insert(0, str(args.f5tts_dir))
    import torch

    from f5_tts.infer.utils_infer import infer_process, load_model, preprocess_ref_audio_text
    from f5_tts.model.utils import get_tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_file = str(args.f5tts_dir / "data" / f"{args.dataset_name}_{'pinyin'}/vocab.txt")
    ckpt = sorted(args.ckpt_dir.glob("model_*.safetensors"))[-1]
    print("checkpoint:", ckpt)

    # use_ema=False: EMA weights of early-stage fine-tunes stay dominated by the pretrained model
    vocos, tokenizer, model, cfg = load_model(
        "F5TTS_v1_Base",
        ckpt_path=str(ckpt),
        vocab_file=vocab_file,
        ode_method="euler",
        use_ema=False,
        device=device,
    )

    if args.ref_wav is None or args.ref_text is None:
        raise SystemExit("--ref_wav and --ref_text are required (clean narrator clip for voice cloning)")
    ref_audio, ref_text = preprocess_ref_audio_text(str(args.ref_wav), args.ref_text, device=device)

    rows = []
    with open(args.metadata_val, encoding="utf-8") as f:
        for line in f.readlines()[1 : args.num_samples + 1]:
            rows.append(line.rstrip("\n").split("|", 1))

    from transformers import pipeline

    asr = pipeline(
        "automatic-speech-recognition",
        model="facebook/mms-1b-all",
        chunk_length_s=10,
        stride_length_s=2,
        device=device,
    )

    mel_spec_kwargs = dict(
        n_fft=1024, hop_length=256, win_length=1024, n_mel_channels=100, target_sample_rate=24000, device=device
    )
    totals = {"cer": 0.0, "wer": 0.0}
    for i, (wav, text) in enumerate(rows):
        gen = infer_process(
            ref_audio,
            ref_text,
            text,
            model,
            tokenizer,
            mel_spec_kwargs=mel_spec_kwargs,
            vocab_file=vocab_file,
            seed=-1,
            nfe_step=16,
            cfg_strength=2.0,
            sway_sampling_coef=-1.0,
            speed=1.0,
            remove_silence=True,
        )
        hyp = asr(gen)["text"]
        totals["cer"] += cer(normalize(text), normalize(hyp))
        totals["wer"] += wer(normalize(text), normalize(hyp))
        print(f"[{i}] ref: {text}")
        print(f"[{i}] hyp: {hyp}")

    n = len(rows)
    print(f"\nCER: {totals['cer'] / n:.4f}  WER: {totals['wer'] / n:.4f}")


if __name__ == "__main__":
    main()