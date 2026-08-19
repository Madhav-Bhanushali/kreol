"""ASR leg for the Gemma voice pipeline.

- `--asr gemma` (default): transcribe via the Gemma audio endpoint
  (GEMMA_AUDIO_BASE_URL / GEMMA_AUDIO_API_KEY). Inputs over ~28s are split by
  the silence/VAD chunker first (30s encoder limit).
- `--asr mms`: fallback ASR via `facebook/mms-1b-all` (explicitly covers `mfe`)
  on a local GPU/CPU — used if Gemma's Morisyen ASR turns out to be weak.

Output: JSON array of per-chunk results {index, start, end, text} plus a
joined full transcript printed to stdout.
"""
import argparse
import base64
import json
import time

import requests

from gemma_env import (
    GEMMA_AUDIO_API_KEY,
    GEMMA_AUDIO_BASE_URL,
    GEMMA_MODEL,
)
from vad_chunk import chunk_file

TRANSCRIBE_PROMPT = (
    "Transcribe this audio clip verbatim, word for word, in exactly the language "
    "spoken in it. Do not add commentary, notes, or translation. Output only the "
    "transcription."
)


def _chat_completion(url, api_key, payload, timeout=180, retries=2):
    headers = {"Authorization": f"Bearer {api_key}"}
    last = None
    for i in range(retries + 1):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                return data["choices"][0]["message"]["content"].strip()
            last = f"HTTP {r.status_code}: {r.text[:400]}"
        except Exception as e:  # noqa: BLE001
            last = str(e)
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"gemma audio request failed: {last}")


def gemma_transcribe_wav_bytes(wav_bytes, api_key=None, base_url=None, model=None, fmt="wav"):
    api_key = api_key or GEMMA_AUDIO_API_KEY
    base_url = base_url or GEMMA_AUDIO_BASE_URL
    model = model or GEMMA_MODEL
    b64 = base64.b64encode(wav_bytes).decode()
    url = f"{base_url}/chat/completions"

    payload_input_audio = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}},
                    {"type": "text", "text": TRANSCRIBE_PROMPT},
                ],
            }
        ],
    }
    payload_audio_url = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": f"data:audio/{fmt};base64,{b64}"}},
                    {"type": "text", "text": TRANSCRIBE_PROMPT},
                ],
            }
        ],
    }
    try:
        return _chat_completion(url, api_key, payload_input_audio)
    except Exception as e1:  # noqa: BLE001
        try:
            return _chat_completion(url, api_key, payload_audio_url)
        except Exception as e2:  # noqa: BLE001
            raise RuntimeError(f"input_audio style failed ({e1}); audio_url style failed ({e2})")


def mms_transcribe_wav(wav_path, lang="mfe", device=None):
    """Fallback ASR leg: facebook/mms-1b-all (covers mfe) via local torch."""
    from vad_chunk import load_wav

    y, sr = load_wav(wav_path, sr=16000)
    return mms_transcribe_waveform(y, sr, lang=lang, device=device)


def mms_transcribe_waveform(y, sr, lang="mfe", device=None):
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2Processor.from_pretrained("facebook/mms-1b-all")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all").to(device)

    inputs = processor(y, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits
    ids = torch.argmax(logits, dim=-1)[0]
    return processor.decode(ids).strip()


def transcribe_file(audio_path, asr="gemma", max_seconds=28.0, top_db=35.0, out=None):
    import io

    import soundfile as sf

    from vad_chunk import load_wav, silence_chunks

    y, sr = load_wav(audio_path, sr=16000)
    chunks = silence_chunks(y, sr, max_seconds=max_seconds, top_db=top_db)
    results = []
    for i, (start, end) in enumerate(chunks, 1):
        t0 = time.time()
        seg = y[int(start * sr): int(end * sr)]
        if asr == "gemma":
            buf = io.BytesIO()
            sf.write(buf, seg, sr, format="WAV")
            text = gemma_transcribe_wav_bytes(buf.getvalue())
        elif asr == "mms":
            text = mms_transcribe_waveform(seg, sr)
        else:
            raise ValueError(f"unknown --asr {asr!r} (use 'gemma' or 'mms')")
        dt = time.time() - t0
        results.append({"index": i, "start": start, "end": end, "seconds": round(end - start, 3),
                        "asr_seconds": round(dt, 2), "text": text})
        print(f"[chunk {i}/{len(chunks)} {start:.2f}-{end:.2f}s ({dt:.1f}s ASR)] {text}")

    full = " ".join(r["text"] for r in results).strip()
    print(f"\nFULL TRANSCRIPT ({len(results)} chunks):\n{full}")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")
    return results, full


def main():
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Transcribe audio with Gemma (or mms-1b-all fallback)")
    ap.add_argument("audio", help="wav (or mp3) file to transcribe")
    ap.add_argument("--asr", choices=["gemma", "mms"], default="gemma")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    ap.add_argument("--max-seconds", type=float, default=28.0)
    ap.add_argument("--top-db", type=float, default=35.0)
    args = ap.parse_args()

    if args.asr == "mms":
        text = mms_transcribe_wav(args.audio)
        print(f"MMS ASR transcript:\n{text}")
        if args.out:
            Path(args.out).write_text(text + "\n", encoding="utf-8")
        return
    transcribe_file(args.audio, asr="gemma", max_seconds=args.max_seconds,
                    top_db=args.top_db, out=args.out)


if __name__ == "__main__":
    main()