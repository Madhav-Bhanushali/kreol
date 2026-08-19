"""Shared config for the Gemma voice pipeline.

Reads KEY=VALUE lines from the repo-root .env (already gitignored). Real
environment variables take precedence over .env entries.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"


def _load():
    if not ENV_PATH.exists():
        return
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load()

GEMMA_TEXT_BASE_URL = os.environ.get("GEMMA_TEXT_BASE_URL", "").rstrip("/")
GEMMA_AUDIO_BASE_URL = os.environ.get("GEMMA_AUDIO_BASE_URL", "").rstrip("/")
GEMMA_TEXT_API_KEY = os.environ.get("GEMMA_TEXT_API_KEY_SMARTTESTING", "")
GEMMA_AUDIO_API_KEY = os.environ.get("GEMMA_AUDIO_API_KEY", "")
GEMMA_MODEL = os.environ.get("GEMMA_MODEL", "google/gemma-4-E4B-it")
ENABLE_GEMMA = os.environ.get("ENABLE_GEMMA", "0") == "1"

# headroom under the 30s encoder limit
MAX_CHUNK_SECONDS = 28.0