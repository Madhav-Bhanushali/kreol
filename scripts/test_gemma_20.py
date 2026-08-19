"""Make N calls to the Gemma text endpoint with varied-language inputs and write
the input/output pairs to a single text file.

Uses the live-demo Kreol-only system prompt (any language in -> Kreol out).
Run from the repo root (reads .env there):
    python scripts/test_gemma_20.py --out gemma_20_test.txt
"""
import argparse
import time
from pathlib import Path

from gemma_env import GEMMA_TIMEOUT
from gemma_reason import generate
from live_demo import RESPONSE_INSTRUCTION, SYSTEM_KREOL_ONLY

SAMPLES = [
    ("fr", "Bonjour, est-ce que tu peux me dire comment cuisiner le riz créole ?"),
    ("en", "Hello, can you tell me about Mauritius?"),
    ("hi", "नमस्ते, आपका दिन कैसा चल रहा है?"),
    ("zh", "你好，请问毛里求斯有什么好吃的？"),
    ("es", "Hola, ¿qué tal el clima en tu país?"),
    ("de", "Hallo, wie geht es dir heute?"),
    ("ar", "مرحبا، كيف حالك؟"),
    ("mfe", "Bonzur, ki ou ti pe fer lot zour?"),
    ("mfe", "Kouma mo kapav resersa en bann resip lor internet?"),
    ("en", "What is the capital of Mauritius?"),
    ("fr", "Peux-tu me raconter une histoire courte ?"),
    ("zh", "你能教我一句毛里求斯克里奥尔语吗？"),
    ("pt", "Olá, como está o tempo hoje?"),
    ("it", "Ciao, come stai?"),
    ("ru", "Привет, как дела?"),
    ("ta", "வணக்கம், எப்படி இருக்கிறீர்கள்?"),
    ("ur", "سلام، آپ کیسے ہیں؟"),
    ("mfe", "Ki to panse lor laplenn Moris?"),
    ("mfe", "Kouma pou al Laport Louis?"),
    ("en", "Can you sing me a song in Kreol?"),
]


def main():
    ap = argparse.ArgumentParser(description="20-call Gemma reply test")
    ap.add_argument("--out", default="gemma_20_test.txt")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    lines = []
    lines.append(f"Gemma 20-call reply test — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"model={__import__('gemma_env').GEMMA_MODEL}  timeout={GEMMA_TIMEOUT}s")
    lines.append("system prompt: " + SYSTEM_KREOL_ONLY.replace("\n", " "))
    lines.append("=" * 78)

    for i, (lang, text) in enumerate(SAMPLES[:args.n], 1):
        user = RESPONSE_INSTRUCTION.format(transcript=text)
        t0 = time.time()
        try:
            reply = generate(SYSTEM_KREOL_ONLY, user)
            status = f"ok ({time.time() - t0:.1f}s)"
        except Exception as e:  # noqa: BLE001
            reply = f"<ERROR> {e}"
            status = f"FAIL ({time.time() - t0:.1f}s)"
        lines.append(f"\n[{i:02d}] {status}  language={lang}")
        lines.append(f"  INPUT : {text}")
        lines.append(f"  OUTPUT: {reply}")
        print(f"[{i:02d}] {status} ({lang}) {text[:40]}...")

    out_path = Path(args.out)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out_path.resolve()}  ({len(lines)} lines)")


if __name__ == "__main__":
    main()