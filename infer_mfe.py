"""Chatterbox mfe inference for quick A/B listening tests.

Usage:
    python infer_mfe.py "Bonzur, kouma ou ye?" \
        --output out.wav --language mfe \
        --exaggeration 0.5 --cfg-weight 0.5 --temperature 0.8

Controls (verified against ChatterboxMultilingualTTS.generate):
  --exaggeration  emotion intensity (library docs range ~0.25-2.0, default 0.5)
  --cfg-weight    classifier-free-guidance strength (default 0.5). NOT a pace
                  knob: logits = cond + cfg*(cond - uncond). Chatterbox has no
                  speech-rate parameter; pacing is set by the model's token count.
  --temperature   sampling randomness (default 0.8; must be > 0)

Long text: the model emits EOS after ~one ~15s utterance (training clips were
~15s), so long text is split into sentences, each synthesized separately and
joined with 0.25s pauses. No hard length cap.
"""
import argparse
import os
import re
import sys

import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import TrainConfig
from src.model import resize_and_load_t3_weights
from src.utils import trim_silence_with_vad
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.models.t3.t3 import T3
from safetensors.torch import load_file

_SENT_RE = re.compile(r"(?<=[.!?…])\s+")
_COMMA_RE = re.compile(r"(?<=[,;:،])\s+")
_MAX_SEG = 300


def split_text(text):
    """Split text into segments the model can complete before EOS.

    The mfe model was trained on ~15s auto-chunked clips, so it emits EOS
    after roughly one utterance (~15s / ~25 tokens-per-sec). Passing a long
    single string truncates to that. Splitting into sentences (and further
    splitting any segment over _MAX_SEG chars on commas) keeps each generation
    short enough to finish properly; segments are joined with short pauses.
    """
    segments = []
    for sent in _SENT_RE.split(text.strip()):
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) <= _MAX_SEG:
            segments.append(sent)
        else:
            parts = _COMMA_RE.split(sent)
            cur = ""
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if cur and len(cur) + 1 + len(part) > _MAX_SEG:
                    segments.append(cur)
                    cur = part
                else:
                    cur = f"{cur} {part}".strip() if cur else part
            if cur:
                segments.append(cur)
    return segments


def synth_segment(eng, text, ref_wav, args):
    wav = eng.generate(
        text=text,
        language_id=args.language,
        audio_prompt_path=ref_wav,
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg_weight,
        temperature=args.temperature,
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
    )
    if isinstance(wav, tuple):
        wav = wav[0]
    return trim_silence_with_vad(wav.squeeze().cpu().numpy(), eng.sr)


def load_merged_engine():
    cfg = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    temp = ChatterboxMultilingualTTS.from_local(cfg.model_dir, device="cpu", t3_model=cfg.t3_model)
    state = temp.t3.state_dict()
    new_config = temp.t3.hp
    new_config.text_tokens_dict_size = cfg.new_vocab_size
    new_t3 = T3(hp=new_config)
    new_t3 = resize_and_load_t3_weights(new_t3, state, new_token_init_row=cfg.fr_token_id)
    del temp, state

    merged_path = os.path.join(cfg.output_dir, "t3_finetuned_merged.safetensors")
    if not os.path.exists(merged_path):
        raise FileNotFoundError(f"Merged checkpoint not found: {merged_path} (run merge_lora.py first)")
    new_t3.load_state_dict(load_file(merged_path), strict=True)

    eng = ChatterboxMultilingualTTS.from_local(cfg.model_dir, device="cpu", t3_model=cfg.t3_model)
    eng.t3 = new_t3
    eng.t3.to(device).eval()
    eng.s3gen.to(device).eval()
    eng.ve.to(device).eval()
    eng.device = device
    return eng


def main():
    ap = argparse.ArgumentParser(description="Chatterbox mfe TTS inference (merged checkpoint)")
    ap.add_argument("text", help="text to synthesize")
    ap.add_argument("--output", default="output.wav", help="output wav path")
    ap.add_argument("--language", default="mfe", help="language id (default mfe)")
    ap.add_argument("--exaggeration", type=float, default=0.5,
                    help="emotion intensity, ~0.25-2.0 (default 0.5)")
    ap.add_argument("--cfg-weight", type=float, default=0.5,
                    help="CFG strength (default 0.5); not a speed/pace control")
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="sampling temperature > 0 (default 0.8)")
    args = ap.parse_args()

    if args.exaggeration < 0.1 or args.exaggeration > 3.0:
        print(f"WARNING: exaggeration {args.exaggeration} is outside the typical 0.25-2.0 range")
    if args.cfg_weight < 0.0 or args.cfg_weight > 2.0:
        print(f"WARNING: cfg-weight {args.cfg_weight} outside typical 0.0-1.5 range")
    if args.temperature <= 0.0:
        print("ERROR: temperature must be > 0")
        return 2

    try:
        eng = load_merged_engine()
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return 1

    cfg = TrainConfig()
    refs = sorted(os.listdir(cfg.wav_dir))
    if not refs:
        print("ERROR: no reference wavs found in dataset wav_dir")
        return 1
    ref_wav = os.path.join(cfg.wav_dir, refs[0])

    try:
        segments = split_text(args.text)
        if not segments:
            print("ERROR: no text to synthesize")
            return 1

        pause = np.zeros(int(eng.sr * 0.25), dtype=np.float32)
        parts = []
        for i, seg in enumerate(segments, 1):
            print(f"[{i}/{len(segments)}] synthesizing: {seg[:60]}{'...' if len(seg) > 60 else ''}")
            parts.append(synth_segment(eng, seg, ref_wav, args))
        audio = parts[0]
        for part in parts[1:]:
            audio = np.concatenate([audio, pause, part])
        audio = trim_silence_with_vad(audio, eng.sr)

        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        sf.write(args.output, audio, eng.sr)
    except Exception as e:
        print(f"ERROR during generation: {e}")
        return 1

    print(f"Generated: {args.output}")
    print(f"  language    : {args.language}")
    print(f"  exaggeration: {args.exaggeration}")
    print(f"  cfg_weight  : {args.cfg_weight}")
    print(f"  temperature : {args.temperature}")
    print(f"  segments    : {len(segments)}")
    print(f"  duration    : {len(audio) / eng.sr:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())