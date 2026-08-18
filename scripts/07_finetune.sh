#!/usr/bin/env bash
set -euo pipefail

# Fine-tune F5-TTS v1 Base on Mauritian Creole.
# Tuned for a single NVIDIA RTX 6000 Ada (48 GB) with bf16.
# Run from anywhere; F5TTS_DIR must point at your F5-TTS checkout (editable install).

F5TTS_DIR="${F5TTS_DIR:-$(realpath ../F5-TTS)}"
DATASET_NAME="${DATASET_NAME:-MFEBSM}"
TOKENIZER="${TOKENIZER:-pinyin}"

cd "$F5TTS_DIR"

accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  src/f5_tts/train/finetune_cli.py \
  --exp_name F5TTS_v1_Base \
  --dataset_name "$DATASET_NAME" \
  --tokenizer "$TOKENIZER" \
  --finetune \
  --learning_rate 1e-5 \
  --batch_size_per_gpu 6400 \
  --batch_size_type frame \
  --max_samples 64 \
  --grad_accumulation_steps 1 \
  --num_warmup_updates 1000 \
  --save_per_updates 1000 \
  --last_per_updates 200 \
  --keep_last_n_checkpoints 3 \
  --logger tensorboard