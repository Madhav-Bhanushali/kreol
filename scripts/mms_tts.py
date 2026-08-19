"""Placeholder output voice for the Gemma pipeline.

Uses the existing pretrained `facebook/mms-tts-mfe` checkpoint unmodified
(see project doc "Dependency note") until the fine-tuned F5-TTS model exists.
"""
import argparse
import time
from pathlib import Path


def synthesize(text, out_path, model_name="facebook/mms-tts-mfe", device=None):
    import soundfile as sf
    import torch
    from transformers import AutoTokenizer, VitsModel

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = VitsModel.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(inputs.input_ids.to(device))
    wav = output.waveform.squeeze().cpu().numpy()
    sr = model.config.sampling_rate
    sf.write(out_path, wav, sr)
    return out_path, sr, wav.size / sr


def main():
    ap = argparse.ArgumentParser(description="TTS with facebook/mms-tts-mfe (placeholder voice)")
    ap.add_argument("--text", default=None)
    ap.add_argument("--text-file", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.text and args.text_file:
        raise SystemExit("give either --text or --text-file, not both")
    text = args.text or Path(args.text_file).read_text(encoding="utf-8").strip()

    t0 = time.time()
    path, sr, dur = synthesize(text, args.out)
    print(f"wrote {path}  ({dur:.2f}s @ {sr}Hz)  in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()