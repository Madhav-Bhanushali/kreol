import argparse
import json
import re
from pathlib import Path

import torch


def norm_word(w: str) -> str:
    return re.sub(r"[^\w]", "", w.lower())


def load_aligner(model_name, device, dtype):
    from ctc_forced_aligner import load_alignment_model

    return load_alignment_model(device, dtype=dtype, model_name=model_name)


def align_chapter(aligner, tokenizer, audio_path, text, language="mfe", batch_size=1):
    from ctc_forced_aligner import (
        generate_emissions,
        get_alignments,
        get_spans,
        load_audio,
        postprocess_results,
        preprocess_text,
    )

    wav = load_audio(str(audio_path), aligner.dtype, aligner.device)
    emissions, stride = generate_emissions(aligner, wav, batch_size=batch_size)
    tokens_starred, text_starred = preprocess_text(text, romanize=True, language=language)
    segments, scores, blank = get_alignments(emissions, tokens_starred, tokenizer)
    spans = get_spans(tokens_starred, segments, blank)
    return postprocess_results(text_starred, spans, stride, scores)


def greedy_match(ref_tokens, words):
    # map each reference token to an aligned word index (greedy forward search)
    out = [None] * len(ref_tokens)
    i = 0
    for k, tok in enumerate(ref_tokens):
        for j in range(i, min(len(words), i + 4)):
            if norm_word(words[j]["text"]) == tok:
                out[k] = j
                i = j + 1
                break
    return out


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Forced-align chapter audio against verse text with facebook/mms-fa. "
            "Input verses.json: { wav_stem: { verse_num: text } }"
        )
    )
    ap.add_argument("verses_json", type=Path, help="data/verses.json")
    ap.add_argument("wav_dir", type=Path, help="data/chapters_clean (or chapters_24k if no music removal)")
    ap.add_argument("--out_dir", type=Path, default=Path("data/alignments"))
    ap.add_argument("--model", default="facebook/mms-fa")
    ap.add_argument("--language", default="mfe")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float16" if torch.cuda.is_available() else "float32")
    args = ap.parse_args()

    with open(args.verses_json, encoding="utf-8") as f:
        chapters = json.load(f)

    aligner, tokenizer = load_aligner(args.model, args.device, getattr(torch, args.dtype))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for stem, verses in chapters.items():
        audio_path = args.wav_dir / f"{stem}.wav"
        if not audio_path.exists():
            print(f"SKIP (no wav): {stem}")
            continue

        verses = {int(k): v for k, v in verses.items()}
        ordered = sorted(verses.items())
        full_text = " ".join(v for _, v in ordered)
        words = align_chapter(aligner, tokenizer, audio_path, full_text, language=args.language)

        ref_tokens = [norm_word(t) for t in full_text.split()]
        word_ids = greedy_match(ref_tokens, words)

        result = {}
        pos = 0
        for num, text in ordered:
            n_words = len(text.split())
            ids = word_ids[pos : pos + n_words]
            pos += n_words
            ids = [i for i in ids if i is not None]
            if len(ids) < max(1, n_words // 2):
                result[str(num)] = {"status": "fail", "text": text, "reason": "few aligned words"}
                continue
            seg_words = [words[i] for i in ids]
            result[str(num)] = {
                "status": "ok",
                "text": text,
                "start": min(w["start"] for w in seg_words),
                "end": max(w["end"] for w in seg_words),
                "score": sum(float(w["score"]) for w in seg_words) / len(seg_words),
                "words": len(seg_words),
            }

        out = args.out_dir / f"{stem}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        ok = sum(1 for v in result.values() if v["status"] == "ok")
        print(f"{stem}: {ok}/{len(ordered)} verses aligned")


if __name__ == "__main__":
    main()