"""Multilingual ASR via faster-whisper with automatic language detection.

Used by the live demo so the user can speak ANY language. The transcript is
handed to Gemma, which is instructed to reply only in Kreol Morisien.
Whisper handles arbitrary-length audio itself (silero VAD inside), so the 30s
Gemma-audio chunking rule does not apply to this leg — only text reaches Gemma.
"""
import argparse
import json
from pathlib import Path

from vad_chunk import load_wav

SIZES = ("tiny", "base", "small", "medium", "large-v3", "large-v3-turbo")


def load_model(model_size="small", device=None, compute_type=None):
    from faster_whisper import WhisperModel

    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe(audio_path, model_size="small", device=None, compute_type=None):
    """Transcribe audio in any language.

    Returns (text, [detected_languages], [per_segment], primary_lang).
    """
    model = load_model(model_size, device=device, compute_type=compute_type)
    y, sr = load_wav(audio_path, sr=16000)
    segments, info = model.transcribe(y, language=None, vad_filter=True)

    per = []
    parts = []
    for s in segments:
        seg_lang = getattr(s, "language", None) or info.language
        per.append({"start": round(s.start, 2), "end": round(s.end, 2),
                    "lang": seg_lang, "text": s.text.strip()})
        parts.append(s.text.strip())
    langs = sorted({p["lang"] for p in per})
    return " ".join(parts).strip(), langs, per, info.language


def main():
    ap = argparse.ArgumentParser(description="Multilingual ASR (whisper, auto language)")
    ap.add_argument("audio")
    ap.add_argument("--model", choices=SIZES, default="small")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    text, langs, per, primary = transcribe(args.audio, model_size=args.model)
    print(f"detected: {primary} | langs={langs}")
    for s in per:
        print(f"  [{s['start']:.2f}-{s['end']:.2f}s {s['lang']}] {s['text']}")
    print(f"\nTRANSCRIPT:\n{text}")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"text": text, "langs": langs, "primary": primary, "segments": per},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()