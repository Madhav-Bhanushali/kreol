"""Gemma text (LLM reasoning) leg for the voice pipeline.

Calls the OpenAI-compatible text endpoint (GEMMA_TEXT_BASE_URL /
GEMMA_TEXT_API_KEY_SMARTTESTING) and returns the generated reply text.
"""
import argparse
import time

import requests

from gemma_env import GEMMA_MODEL, GEMMA_TEXT_API_KEY, GEMMA_TEXT_BASE_URL, GEMMA_TIMEOUT


def generate(system, user, api_key=None, base_url=None, model=None, max_tokens=512,
             temperature=0.7, timeout=None, retries=1):
    api_key = api_key or GEMMA_TEXT_API_KEY
    base_url = base_url or GEMMA_TEXT_BASE_URL
    model = model or GEMMA_MODEL
    timeout = timeout or GEMMA_TIMEOUT
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    last = None
    for i in range(retries + 1):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
            last = f"HTTP {r.status_code}: {r.text[:400]}"
        except Exception as e:  # noqa: BLE001
            last = str(e)
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"gemma text request failed: {last}")


def main():
    ap = argparse.ArgumentParser(description="Ask the Gemma text endpoint something")
    ap.add_argument("--system", default=None)
    ap.add_argument("--user", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    out = generate(args.system, args.user, max_tokens=args.max_tokens, temperature=args.temperature)
    print(out)


if __name__ == "__main__":
    main()