"""End-to-end glue pipeline: audio in -> (Gemma or MMS ASR) -> transcript ->
Gemma reasoning -> response text -> TTS -> audio out.

Stages are timed and printed so per-stage latency can be reviewed (see the
project doc's final To-do: end-to-end latency check).
"""
import argparse
import json
import time
from pathlib import Path

from gemma_env import GEMMA_MODEL, GEMMA_TEXT_API_KEY, GEMMA_TEXT_BASE_URL
from gemma_reason import generate

DEFAULT_SYSTEM = (
    "You are a helpful voice assistant for a Mauritian Creole (Kreol Morisien) "
    "speaker. The user speaks to you in voice; you are given their transcript. "
    "Respond helpfully, concisely, in the same language they used. When unsure, "
    "ask a short clarifying question in that language."
)

RESPONSE_INSTRUCTION = (
    "\n\nUser's spoken message (transcribed):\n{transcript}\n\n"
    "Respond as yourself, directly to this message."
)


def run_pipeline(audio_path, out_wav, asr="gemma", system=None, lang="mfe",
                 max_seconds=28.0, top_db=35.0, transcript_out=None):
    from gemma_asr import transcribe_file

    times = {}
    t0 = time.time()
    results, transcript = transcribe_file(
        audio_path, asr=asr, max_seconds=max_seconds, top_db=top_db, out=transcript_out
    )
    times["asr"] = time.time() - t0

    if not transcript:
        raise SystemExit("empty transcript — nothing to respond to")

    system = system or DEFAULT_SYSTEM
    user = RESPONSE_INSTRUCTION.format(transcript=transcript)
    t0 = time.time()
    reply = generate(system, user)
    times["reason"] = time.time() - t0
    print(f"\nREPLY:\n{reply}")

    t0 = time.time()
    from mms_tts import synthesize

    path, sr, dur = synthesize(reply, out_wav)
    times["tts"] = time.time() - t0
    times["total"] = sum(times.values())

    print(f"\nwrote {path} ({dur:.2f}s @ {sr}Hz)")
    print(f"latency: asr={times['asr']:.1f}s reason={times['reason']:.1f}s "
          f"tts={times['tts']:.1f}s total={times['total']:.1f}s")
    return {"transcript": transcript, "reply": reply, "wav": str(path), "latency": times}


def main():
    ap = argparse.ArgumentParser(description="Gemma voice pipeline (ASR -> reason -> TTS)")
    ap.add_argument("--audio", required=True, help="input audio (wav/mp3)")
    ap.add_argument("--out", required=True, help="output wav path")
    ap.add_argument("--asr", choices=["gemma", "mms"], default="mms",
                    help="ASR leg: 'mms' (mms-1b-all, decided fallback — Gemma ASR underperformed on Morisyen) or 'gemma'")
    ap.add_argument("--system", default=None)
    ap.add_argument("--lang", default="mfe")
    ap.add_argument("--max-seconds", type=float, default=28.0)
    ap.add_argument("--top-db", type=float, default=35.0)
    ap.add_argument("--transcript-out", default=None)
    args = ap.parse_args()

    run_pipeline(
        args.audio, args.out, asr=args.asr, system=args.system, lang=args.lang,
        max_seconds=args.max_seconds, top_db=args.top_db, transcript_out=args.transcript_out,
    )


if __name__ == "__main__":
    main()