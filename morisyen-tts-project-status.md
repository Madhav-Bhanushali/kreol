# Mauritian Creole (Morisyen) TTS — Project Status

## PIVOT NOTICE (read first)
Plan changed. Everything previously active is now on hold. New active task: **fine-tune Chatterbox Multilingual V3 on the Bible dataset, warm-starting from its native French capability, to get emotion-controllable Morisyen speech.**

---

## ⏸️ On hold — F5-TTS fine-tuning plan

Superseded by the Chatterbox v3 plan below — same underlying Bible audio/text data gets reused, different model/architecture.

<details>
<summary>Click to expand: paused F5-TTS plan</summary>

- Base model was F5-TTS (flow-matching, DiT), vanilla checkpoint, no warm-start (dropped for time).
- License problem that motivated this pivot: F5-TTS's pretrained checkpoint is CC-BY-NC (inherited from Emilia training data) — any fine-tune stays non-commercial.
- Fine-tuning pipeline (`finetune_cli.py`, `--tokenizer char`, etc.) was fully worked out — see prior conversation history if this path is ever revisited.

</details>

## ⏸️ On hold — Gemma audio pipeline

Set aside per pivot. Was: audio-in Gemma (ASR + reasoning) → TTS → audio out, using `facebook/mms-tts-mfe` as placeholder voice. Revisit once the Chatterbox fine-tune produces a real Morisyen checkpoint to plug into the TTS leg of that pipeline.

---

## ✅ Active task — COMPLETE (final artifact delivered Aug 20, 2026)

**Final results (production deliverable):** `mfe_output/final/` — standalone merged checkpoint `t3_finetuned_merged.safetensors` (2.14 GB) + pre-merge 10-epoch LoRA adapter (`adapter/`) + `README.md`; integrity-verified (sha256 vs backup, `[mfe]` row bit-identical merged↔adapter for text_emb & text_head).

- **Base model:** Chatterbox V3 (`t3_model="v3"`), MIT-licensed.
- **Warm-start:** `[mfe]` row (id 2454) copied from `[fr]` (id 634), then fine-tuned.
- **Data:** WorldSpeech `mfe_mu`, CER-filtered (<0.3) → 9,645 clips / ~39.8 h.
- **Training:** 10 epochs (3,010 steps, ~22 min), LoRA r=32 on q/k/v/o only, rest frozen, bf16.
- **Eval (mms-1b-all, mfe):** held-out 80 texts → **CER 0.146 / WER 0.239**; ASR ceiling on real audio 0.122/0.226; merged sanity (n=10) 0.155/0.235.
- **Not adopted:** 15-epoch extended run (overfit — eval worsened to 0.153/0.260) — kept only in `extended_15ep/` for reference.

<details>
<summary>Full build detail and history below</summary>

### Why this approach
1. **License is finally clean.** Chatterbox's code and pretrained checkpoints are MIT licensed — unlike every other base model considered so far (F5-TTS, MMS-TTS all CC-BY-NC). A fine-tune from this checkpoint is not automatically restricted to non-commercial use the way the earlier paths were.
2. **French warm-start is free** — French is one of Chatterbox V3's 23 native base languages, so there's no need to hunt down a separate community French checkpoint (unlike the F5-TTS plan, which needed `RASPIAUDIO/F5-French-MixedSpeakers-reduced`). Since Kreol Morisien vocabulary is heavily French-derived, starting from a model that already has French phonetics/prosody should meaningfully reduce how much the fine-tune needs to learn from nothing — same warm-start logic used for Marathi←Hindi and the existing `chatterbox-indic-lora` project's Brahmic warm-starts.
3. **Native emotion control** — Chatterbox V3 is the first open-source TTS model with built-in emotion exaggeration control (`exaggeration` parameter, 0.25–2.0, default 0.5), independent of language. This is architecturally already there in the base model; fine-tuning needs to preserve and extend it into Morisyen rather than build it from scratch.

### Architecture recap
- **T3** (0.5B Llama-based token generator): text/audio-token transformer, decoder-only, predicts speech tokens conditioned on text + language ID + speaker embedding + the exaggeration/emotion conditioning.
- **S3Gen**: diffusion-based vocoder, converts speech tokens to waveform.
- Same brain/vocoder split established earlier in this project: **T3 is what gets fine-tuned for the new language; S3Gen stays untouched.**

### Warm-start mechanism (same pattern as `reenigne314/chatterbox-indic-lora`, applied to French → Morisyen instead of Hindi → other Indic languages)
1. Start from the official base checkpoint directly — no separate French model needed:
   ```python
   from chatterbox.mtl_tts import ChatterboxMultilingualTTS

   model = ChatterboxMultilingualTTS.from_pretrained(device="cuda", t3_model="v3")
   ```
2. **Tokenizer**: Morisyen is Latin-script, same alphabet family as French — check whether any accented characters used in Kreol Morisien orthography are missing from the existing vocab (likely minimal gaps, unlike the Devanagari/Tamil-script extensions needed for Indic languages). Extend only if needed; mean-init any genuinely new tokens.
3. **Language embedding**: add a new `mfe` row to the language-ID embedding table. **Warm-start it as a copy of the `fr` (French) row**, not random init — this is the crux of the whole warm-start strategy, directly transferring whatever French-phonetics knowledge the base model already has into the new language's starting point.
4. **LoRA fine-tune**, not full fine-tune — same reasoning as before: protects the other 22 languages from catastrophic forgetting. Target the same modules as the reference Indic LoRA project: `q/k/v/o` attention projections in T3 (their config used rank-32, ~7.8M/544M trainable params — a reasonable starting point here too).
5. Freeze S3Gen and the speaker encoder entirely — untouched, per the brain/vocoder split.

### Dataset: WorldSpeech `mfe_mu` (replaces Bible audio plan)

**Switched data source** — using `disco-eth/WorldSpeech`, config `mfe_mu`, instead of the Bible recordings from the paused plan.

- **44.3 hours** of aligned Mauritian Creole speech, quality score 3.45 (DNSMOS-based).
- **Source**: Mauritian National Assembly parliamentary proceedings — public record, not a copyrighted third-party religious translation. Resolves the unconfirmed Bible-text-license question that was an open item in the paused plan.
- **Format already matches what's needed** — no separate forced-alignment step required. Each row already comes as: 24kHz audio, human-provided transcript, aligned ASR transcript, CER, WADA-SNR, DNSMOS-P.835 quality scores.
  ```python
  from datasets import load_dataset

  ds = load_dataset("disco-eth/WorldSpeech", "mfe_mu", split="train")
  row = ds[0]
  wav = row["audio"]["array"]
  sr = row["audio"]["sampling_rate"]
  text = row["human_transcript"]
  cer = row["cer"]
  ```
  Filter on `cer` before using a row for training — same principle as the alignment-confidence filtering that was planned for the Bible data, just already computed for you here.
- **Honest limitation, same shape as before, different source**: WorldSpeech's own paper notes its speaking style is closer to formal prepared speech than to spontaneous conversation, skewed toward parliamentary/broadcast register. This does **not** solve the flat-emotional-range concern — parliamentary proceedings are still a formal register, just a different one than liturgical reading. Don't expect this switch alone to unlock wide `exaggeration` range for Morisyen.
- Bible audio previously obtained is no longer the primary path — keep it around only if WorldSpeech's 44.3h turns out insufficient after a first training pass.

### Toolkit
Same fine-tuning toolkit already scoped earlier in this project: **`gokhaneraslan/chatterbox-finetuning`** (Standard mode, not Turbo — LJSpeech-format dataset, `is_lora=True`, `--tokenizer` handling as described in its repo docs).

### Results from first training run
- **Training**: 10 epochs, 21.8 min, loss 6.85→4.10, adapter verified (12.89M params, `mfe` row anchored to `fr`).
- **`mfe` quality — strong**: CER 0.146 / WER 0.239, essentially at the ASR's own accuracy ceiling on real recordings (CER 0.122 / WER 0.226). 2x better than base+fr baseline (WER 0.481). This part of the plan worked as designed.
- **French regression — observed, but NOT a blocker**: base French WER 5.1% → adapter 63.3%. LoRA + frozen embeddings did not preserve French as hoped (no French replay data was mixed in). **This project only needs mfe output quality; French/other-language preservation was never an actual requirement → mitigation CANCELLED.**

### ⏸️ To do — French regression mitigation (CANCELLED — not an actual requirement)
<details>
<summary>Click to expand: cancelled mitigation plan (kept for reference, do not execute)</summary>

- [ ] **Check epoch 2–3 checkpoints first** (saved every 100 steps during the completed run) — evaluate French WER on these earlier checkpoints. If regression is much lower there while `mfe` CER/WER stays reasonably close to the final-epoch numbers above, this may be usable without a full retrain. *(Note: checkpoints 600/900 were already rotated out — only 2700–3020 remain — so this path was moot.)*
- [ ] **If epoch 2–3 isn't good enough, retrain with French replay**: mix 5–10% French text/audio examples into the `mfe` training batches (matching the general "replay old-language data" step from the architecture doc). Keep all other config identical (same LoRA rank/targets, same `mfe`←`fr` embedding warm-start) so this is a controlled comparison against the first run.
- [ ] Re-run both eval legs after the mitigation run: `mfe` CER/WER via `mms-1b-all` (same as before), and French regression WER — confirm French recovers close to the 5.1% baseline while `mfe` doesn't regress much from 0.146/0.239.

</details>

### To do — next steps (current)
- [x] **Merge LoRA into a standalone checkpoint** (`merge_lora.py`), baking the first run's adapter (`mfe_output/new_lang_adapter`) into the base T3 → single `.safetensors`. Use the existing adapter as-is, do not retrain. *(Done Aug 20: `mfe_output/t3_finetuned_merged.safetensors` 2.14GB; verified text_emb/text_head (2455,1024) and merged `[mfe]` row == adapter row exactly. merge_lora.py was adapted to V3: chatterbox imports + `from_local(t3_model="v3")` + fr-copy resize init.)*
- [x] **Exaggeration listen-test** on the merged model: generate the same test sentences at `exaggeration` 0.3 / 0.5 / 0.8 / 1.2, one separate wav per setting, for by-ear comparison. *(Done Aug 20: 4 sentences × 4 exagg = 16 wavs at `mfe_output/merged_eval/exagg_*.wav` on the box, also copied locally to `exagg_samples/`.)*
- [x] **Quick sanity check**: mfe CER/WER via `mms-1b-all` on a few outputs from the merged standalone checkpoint — confirm it still matches the adapter numbers (~CER 0.146 / WER 0.239), i.e. merging changed nothing. *(Done Aug 20, n=10 held-out texts, merged model: CER 0.155 / WER 0.235 — matches adapter 0.146/0.239 within sampling noise.)*

### 🧪 Experiment — extended training: resume from 10-epoch adapter, +5 epochs (comparison only — NOT adopted yet)
**Status: results in (in progress → decision pending).** Separate from the 10-epoch run below; original production adapter + merged checkpoint untouched.

- [x] **Backup current state before the experiment** → `mfe_output/backup_10ep/` (verified, sha256 match): `t3_finetuned_merged.safetensors`, full `adapter/` (the 10-epoch `new_lang_adapter` used by merge_lora.py), `config.py`, `trainer_state.json` + `loss_curve_10ep.csv` (120-point loss log 6.85→4.07), `tensorboard_runs/`, and `train.log`.
- [x] **Resume training** (`resume_train.py`): loaded the trained 10-epoch adapter (`PeftModel.from_pretrained(..., is_trainable=True)`) instead of fresh LoRA init; all settings identical to first run — LoRA r=32/α=64, targets q/k/v/o, modules_to_save text_emb/text_head, `mfe` row kept warm-started from `fr`, same WorldSpeech `mfe_mu` CER-filtered dataset (reused preprocessed `.pt`), batch 16 × accum 2, lr 1e-4, bf16, no grad-checkpointing. No French replay (mitigation cancelled). Output to **`mfe_output/extended_15ep/`** (original untouched). *Note: fresh optimizer/scheduler → LR re-annealed 1e-4→0 across the 5 epochs.*
- [x] **Run + report loss curve** (5 epochs = 1,505 steps, ~10.5 min, finished Aug 20):

  | step | 10-epoch run | 15-epoch run |
  |------|-------------|--------------|
  | 25   | 6.853       | **4.157**    |
  | 250  | 4.769       | 4.094        |
  | 500  | 4.530       | 4.064        |
  | 750  | 4.415       | 4.040        |
  | 1000 | 4.334       | 4.021        |
  | 1250 | 4.240       | 3.963        |
  | 1500 | 4.225       | 3.973        |
  | 3000 | 4.072       | —            |
  | min  | 4.061       | 3.951        |

  Verdict: **loss is still genuinely moving** (4.16 → 3.97 over the 5 epochs, −0.19) but the curve is flattening — last 100 steps ~4.02→3.97 with mild noise. Not flat, not noisy-plateau, but the per-epoch gain is much smaller than in the first run's early epochs.
- [x] **Eval on same 80 held-out texts (identical generation params, mms-1b-all mfe):**
  - Original 10-epoch adapter: **CER 0.146 / WER 0.239**
  - Extended 15-epoch adapter: **CER 0.153 / WER 0.260** → slightly *worse* on held-out speech despite lower training loss (mild overfitting signature).
- [x] **Nothing merged/replaced** — production checkpoint remains the 10-epoch adapter + `t3_finetuned_merged.safetensors`. Extended artifacts live only in `extended_15ep/` (adapter + checkpoints + eval/metrics.json + loss_curve_15ep.csv).
- **Decision needed:** loss still declining ⇒ per the run-more rule, **confirm with me before running additional epochs**; eval suggests we're past the generalization sweet spot (overfitting), so I'd recommend *not* extending further unless a specific artifact needs it.

### Completed (first training run)
- [x] Load `disco-eth/WorldSpeech` config `mfe_mu` via `datasets`; inspect actual hour count / row count and `cer`/quality-score distribution.
- [x] Filter rows by CER threshold to drop poorly-aligned segments.
- [x] Convert filtered rows into LJSpeech-format dataset (`wavs/` + `metadata.csv`) for the fine-tuning toolkit.
- [x] Load base Chatterbox V3 checkpoint (`t3_model="v3"`), inspect tokenizer vocab.
- [x] Add `mfe` language embedding row, initialized as a copy of the `fr` row.
- [x] Configure LoRA (rank-32, target `q/k/v/o` in T3, freeze S3Gen + speaker encoder + all other language rows).
- [x] Run fine-tune via `gokhaneraslan/chatterbox-finetuning`.
- [x] Evaluate `mfe` WER/CER against `facebook/mms-1b-all` — strong result, see "Results" above.
- [x] Regression check on French — **failed**, see "Results" above, mitigation now in progress.

</details>

---

## Open questions
- Bible text license terms — not yet confirmed.
- Whether to pursue non-liturgical Morisyen audio for genuine emotion-range coverage, or accept a limited emotional range as a known constraint of this data source for now.
