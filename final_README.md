# Morisyen (Mauritian Creole) TTS — Chatterbox V3 fine-tune (FINAL artifact)

This is the production deliverable for the Morisyen TTS fine-tune. Trained with the
`gokhaneraslan/chatterbox-finetuning` toolkit (Standard mode, LoRA).

## Files
- `t3_finetuned_merged.safetensors` — standalone merged checkpoint (2.14 GB, 292 tensors).
  LoRA baked into the base T3. Ready for inference; no PEFT step needed.
- `adapter/` — the pre-merge 10-epoch LoRA adapter (12,892,160 params; r=32, alpha=64,
  targets q/k/v/o, modules_to_save text_emb/text_head). Kept alongside the merged version
  in case the adapter form is ever needed separately.

## Training
- Base model: Chatterbox V3 multilingual (`t3_model="v3"`), MIT-licensed code + weights.
- Warm-start: new `[mfe]` language row (vocab id 2454) initialized as a copy of the
  `[fr]` row (id 634), then fine-tuned (not random init).
- Data: WorldSpeech `mfe_mu` (Mauritian National Assembly proceedings), CER-filtered
  (< 0.3) → 9,645 clips / ~39.8 h / 24 kHz.
- Fine-tune: 10 epochs (3,010 steps, ~22 min); LoRA rank 32 on q/k/v/o attention
  projections only; S3Gen + speaker encoder + all other language rows frozen; bf16;
  batch 16 × grad-accum 2; lr 1e-4; no gradient checkpointing (V3 T3).
- Vocabulary: 2455 tokens (2454 base + new `[mfe]`).

## Evaluation (held-out, mfe)
- Adapter, mms-1b-all, 80 held-out texts: **CER 0.146 / WER 0.239**
- ASR ceiling on real recordings (n=25): **CER 0.122 / WER 0.226** → the adapter sits at
  the ASR's own accuracy limit.
- Merged checkpoint sanity (n=10): CER 0.155 / WER 0.235 — matches adapter within
  sampling noise; merging changed nothing.

## Note
A 15-epoch extended run (resume from this adapter, +5 epochs) was tested and **REJECTED
for overfitting**: held-out eval worsened to CER 0.153 / WER 0.260 despite lower training
loss. This 10-epoch artifact is the production model.

## Artifact integrity
Verified against `backup_10ep/`: sha256-identical copies; the merged `[mfe]` row is
bit-identical to the adapter `[mfe]` row for both text_emb and text_head.

## Usage
- Inference: `infer_mfe.py` on the training box (defaults to `final/`).
- Manual: load `t3_finetuned_merged.safetensors` as the T3 state_dict with vocab size
  2455 (see `merge_lora.py` / `eval_mfe.py` loaders in the training repo).