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

## 🎯 Active task: Fine-tune Chatterbox V3 on Morisyen Bible audio, French warm-start, emotion-aware

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

### To do
- [x] Load `disco-eth/WorldSpeech` config `mfe_mu` via `datasets`; inspect actual hour count / row count and `cer`/quality-score distribution.
- [x] Filter rows by CER threshold (e.g. CER < 0.2–0.3, matching the dataset paper's own quality filtering convention) to drop poorly-aligned segments.
- [x] Convert filtered rows into LJSpeech-format dataset (`wavs/` + `metadata.csv`) for the fine-tuning toolkit.
- [x] Load base Chatterbox V3 checkpoint (`t3_model="v3"`), inspect tokenizer vocab for any missing Morisyen-specific characters.
- [x] Add `mfe` language embedding row, initialized as a copy of the `fr` row (not random).
- [x] Configure LoRA (rank-32 starting point, target `q/k/v/o` in T3, freeze S3Gen + speaker encoder + all other language rows).
- [x] Run fine-tune via `gokhaneraslan/chatterbox-finetuning` (done Aug 20: 10 epochs / 3010 steps / 21.8 min, loss 6.85→4.10, adapter + checkpoints saved).
- [x] Evaluate: WER/CER against `facebook/mms-1b-all` transcription of generated audio; listen-test at a couple of `exaggeration` settings to see whether emotion control transfers at all to `mfe`, even if range is limited by the formal-register training data.
- [x] Regression check: confirm French (and a couple of other original 23 languages) still generate correctly after the `mfe` fine-tune — LoRA + frozen embeddings should protect this, but verify rather than assume.

### Status notes (append-only)
- **Data-source correction:** `mfe_mu` is **not** National Assembly proceedings — every one of the 10,237 rows has `source = jw_bible_kreol_morisien` (the JW Bible in Kreol Morisien, copyright-restricted translation). User confirmed to proceed with it anyway. The license question is therefore still open, same shape as the old Bible plan; attribution/non-commercial constraints likely apply.
- **Dataset inspection (md#1):** 10,237 train rows / 42.1h total audio / all 24 kHz / clips ~15s (constant; min 3.6s). CER: min 0.0, p10 0.095, median 0.191, p90 0.282, max 0.498, no NaN. Test split: 539 rows. Actual quality columns are `snr` + `dnsmos_sig/bak/ovr/p808` (not `wada_snr`/`dnsmos_p835` as the md listed).
- **CER filter (md#2):** chose **CER < 0.3** → keeps 9,681 rows / 39.8h (95%); CER < 0.2 would keep 22.5h, CER < 0.25 keeps 32.0h. Re-runnable via `--cer`.
- **LJSpeech build (md#3):** `scripts/build_ljspeech_dataset.py` → `data/worldspeech_mfe_ljspeech/` (`wavs/` 24 kHz mono PCM-16 + `metadata.csv` as `id|human_transcript|human_transcript`). Workaround: `datasets` 5.0.1 decodes audio via `torchcodec`, whose binary mismatches the box's torch 2.8 (undefined symbol) — the script loads the audio column as raw bytes and decodes with `soundfile` instead. HF token now in `.env` (gitignored) on both machines.
- **Chatterbox is a new toolchain** — not yet installed anywhere; the box has space now (198G free after the user's cleanup), so model/dataset downloads go to the box.
- **Token/vocab check (md#4):** the mfe_mu transcripts use only 24 ASCII letters (a–z minus j/q) — **zero accented characters**, a pure ASCII subset. The V3 multilingual grapheme tokenizer (`grapheme_mtl_merged_expanded_v1.json`, **2,454 tokens**) already contains every letter, all French accented chars, and both apostrophe forms → **no vocab extension needed**. No `<mfe>` language token exists yet (that's md#5).
- **Toolkit note (md#6/7 prep):** `gokhaneraslan/chatterbox-finetuning` Standard mode = Llama-based T3 with the 2,454-token grapheme tokenizer; LoRA adds `lora_r/alpha`, target modules `c_attn/c_proj/c_fc/spkr_enc` (Turbo-style names), saves `text_emb`/`text_head`. Its "new language" path extends the *text* vocab; adding a dedicated `mfe` language-ID row (the md's copy-from-`fr` plan) will likely need a small custom patch on top. PyPI `chatterbox-tts` 0.1.7 is V2-only — V3 (`t3_model="v3"`) requires the GitHub source install (in progress in a dedicated `/root/bonsai/chatterbox_venv`; torch bumped to 2.8.0+cu128 for the Blackwell GPU).
- **LJSpeech dataset built (md#3):** 9,681 rows / 39.8h / 6.5GB at `/root/bonsai/kreol/data/worldspeech_mfe_ljspeech/` (`wavs/` 24 kHz mono PCM-16, `metadata.csv` `id|human_transcript|human_transcript`, 0 skipped). Verified: 15s clips, mono, 24 kHz.
- **V3 checkpoint loads (md#4):** `ChatterboxMultilingualTTS.from_pretrained(t3_model="v3")` works from the GitHub-source install (`/root/bonsai/chatterbox_venv`, torch 2.8.0+cu128). T3 = Llama_520M, `is_multilingual`, `text_tokens_dict_size=2454`, `speaker_embed_size=256`, `emotion_adv=True`.
- **How the language is conditioned (md#5 mechanism):** NOT a separate embedding table — the language is prepended as a token, e.g. `[fr]bonzur kouma ou ye`, embedded by `text_emb` ((2454,1024)); `text_head` is also (2454,1024). The 23 languages each have a `[xx]` token (`[fr]`=id 634, `[en]`=708, …). → adding Morisyen = add a `[mfe]` token (vocab 2454→2455), grow `text_emb`/`text_head` to 2455 rows, init the new row as a **copy of the `[fr]` row** (id 634). This maps cleanly onto the finetuning toolkit's `new_vocab_size` + `lora_modules_to_save=["text_emb","text_head"]` path (just needs fr-copy init instead of mean-init for the new token).
- **Dataset text cleanup (before md#5/6):** user flagged possible text/audio mismatch. Audit found: transcripts are genuinely contiguous with audio (ASR large-v3 self-match 85–94%; whisper reads the audio as French — Kreol isn't a whisper language, so it can't score Kreol reliably), BUT text was auto-chunked independently: ~40% of rows contained literal `\r\n`/`\n` escape sequences, 95% of clips start mid-utterance. Cleaned `metadata.csv` (strip `\r`/`\n`/`\t`, collapse ws, tidy punctuation) and dropped 36 rows (both members of 18 adjacent pairs whose transcripts overlapped ≥40 chars = duplicated speech across the boundary). **9,681 → 9,645 rows / 39.8h.** Original preserved as `metadata_orig.csv` (9,681). Remaining ~1% boundary noise is negligible for LoRA training.
- **Toolkit adapted to V3 (md#5/6, done + verified):** `gokhaneraslan/chatterbox-finetuning` Standard mode is built around the **English (V1) base** (`t3_cfg.safetensors`, vendored V2-only model, mean-init, no language token). Adapted it to **Chatterbox V3 multilingual**: imports redirected from `src.chatterbox_*` to the installed V3 `chatterbox` package; Standard-mode engine = `ChatterboxMultilingualTTS.from_local(..., t3_model="v3")`; `resize_and_load_t3_weights(..., new_token_init_row=634)` copies the `[fr]` row into the new `[mfe]` row (both `text_emb` + `text_head`); tokenizer extended `[mfe]`→id 2454 (vocab 2455); `SUPPORTED_LANGUAGES` += `mfe` in the installed `mtl_tts.py` (needed for `generate(language_id="mfe")`); preprocessing prepends the language token via `text_to_tokens(text, language_id="mfe")` and honors `CB_PREPROC_DEVICE` (default cuda); `check_pretrained_models` now looks for `ve.pt/t3_mtl23ls_v3.safetensors/s3gen.pt/conds.pt/grapheme_mtl_merged_expanded_v1.json`. Config: `lora_r=32, lora_alpha=64`, targets `["q_proj","k_proj","v_proj","o_proj"]` (+ `modules_to_save=["text_emb","text_head"]`), `new_vocab_size=2455`, `fr_token_id=634`, `language_id="mfe"`, paths point at the cleaned LJSpeech dataset. Model files (ve.pt, t3_mtl23ls_v3 2.1GB, s3gen.pt, conds.pt, extended tokenizer) copied into `pretrained_models/`. NOTE: earlier status note said Standard-mode LoRA targets are Turbo-style `c_attn/...` — that is wrong for Standard mode; it's `q_proj/k_proj/v_proj/o_proj` (verified), and "other language rows stay frozen" for free because mfe texts never contain other `[xx]` tokens (only row 2454 + grapheme tokens get gradients).
- **End-to-end smoke test PASSED (md#5/6 verified, no training run):** tokenizer 2455 + `[mfe]`→2454; V3 engine loads; new `text_emb`/`text_head` (2455,1024) with **mfe row == fr row** (≠ mean); LoRA r=32 → **12.89M trainable params (2.35%)** = ~7.9M q/k/v/o LoRA + ~5M text_emb/head (matches the Indic reference's ~7.8M backbone). Mini preprocessing (3 clips, CPU) produces `.pt` with `text_tokens[0]==2454`, speech 376, spk 256, prompt 75. Full 3-clip **Trainer loop ran on transformers 5.2.0 + peft 0.20.0** (loss 10.69, grad_norm 13.32) and saved the LoRA adapter + resized embeddings. Dependencies added to `/root/bonsai/chatterbox_venv`: peft 0.20.0, tensorboard, silero-vad, soundfile. CPU preproc is ~13s/clip (9,645 → ~35h), so full preprocessing + training both wait for GPU.
- **md#7 ready to run (when GPU free):** `cd /root/bonsai/chatterbox-finetuning && /root/bonsai/chatterbox_venv/bin/python train.py` — `preprocess=True` preprocesses on GPU (CB_PREPROC_DEVICE default cuda) then trains. Adapter saves to `mfe_output/new_lang_adapter` (LoRA + resized embeddings). batch_size 16 / grad_accum 2 / lr 1e-4 / 10 epochs (tunable).
- **md#7 TRAINING RUNNING (Aug 20, started 10:08 IST):** full dataset preprocessed on GPU (9,645/9,645 clips; ~13.5 clips/sec → ~12 min, not the feared 35h). First launch crashed at Trainer init: V3 `T3` has **no gradient-checkpointing support** (`'T3' object has no attribute 'gradient_checkpointing_enable'`); the smoke test had skipped GC. Fixed: `gradient_checkpointing=False` in `train.py` (no VRAM need on 96GB) + wrapper GC methods are now no-ops; preprocess skip-guard moved to top of loop so restarts skip done clips. Relaunched 09:55, training began 10:08. **Checkpoints & logs:** `save_steps=100, save_total_limit=5` (keep last 5, ~2.2GB full-model each), `logging_strategy="steps"` logging_steps=25, TensorBoard → `mfe_output/runs/`, console → `kreol/logs/train.log`. Loss curve: 6.85 (st25) → 4.67 (st325), steadily decreasing; grad_norm logged. Speed ≈ 100 optimizer steps/min → **~3 min/epoch**, 10 epochs ≈ 30–40 min total. Run supervised via tmux session `mfe_train` (`kreol/scripts/run_mfe_train.sh`).
- **md#7 TRAINING COMPLETE (Aug 20, 10:30 IST, rc=0):** 10 epochs / 3,010 optimizer steps in **21.8 min** (train_runtime 1307s, train_loss 4.34). Adapter saved to `mfe_output/new_lang_adapter/` and verified: `adapter_config.json` = r=32/alpha=64 on q/k/v/o + `modules_to_save=["text_emb","text_head"]`; `text_emb`/`text_head` are (2455,1024); **12,892,160 params** exactly as planned; 240 LoRA matrices (30 layers × 4 proj × A+B). Checkpoints (last 5: 2700–3020, full-model ~2.2GB each) + TensorBoard retained. On "is there a French LoRA at startup": **no** — `init_lora_weights=true` = LoRA starts at zero (standard PEFT init); the French warm-start is the `[mfe]` embedding row copied from `[fr]` (verified: started exact copy; after training `dist(mfe,fr)=0.164` vs `dist(fr,en)=0.771`). Loss 6.85 start < random baseline ln(2454)≈7.8 → base already predicts acoustic tokens well; 4.10 plateau is in-line.
- **md#8 EVALUATION COMPLETE (Aug 20):** `eval_mfe.py` → 80 held-out **test-split** texts (CER<0.3, never seen in training), generated with adapter+mfe / base+fr / adapter+fr (same mfe reference prompt), scored by `facebook/mms-1b-all` (mfe adapter). Results: **adapter_mfe CER 0.146 / WER 0.239**; base+fr CER 0.138 / WER 0.481; adapter+fr CER 0.130 / WER 0.231. **ASR ceiling on real ground-truth JW audio (n=25): CER 0.122 / WER 0.226** → the fine-tuned model is essentially *at the ASR's own accuracy limit* (ΔCER +0.024, ΔWER +0.013), i.e. highly intelligible fluent Morisyen (spot-checked transcriptions read as clean Kreol). base+fr WER 0.481 (2× worse) confirms the adapter is doing the work. Eval bugs fixed along the way: (1) `torchaudio` import scope, (2) French not in mms-1b-all (high-resource langs excluded) → used `jonatasgrosman/wav2vec2-large-xlsr-53-french` for the French check, (3) `batch_decode` needs a 2D batch tensor (1D input decoded only the first `<pad>` id → false CER 0.98 for everything). Generated wavs for listening: `mfe_output/eval/wavs/*.wav` (adapter_mfe), `*_basefr.wav`, `*_adapterfr.wav`, `mfe_output/eval/fr/*.wav`.
- **md#9 REGRESSION CHECK COMPLETE (Aug 20):** **French regressed** — 20 French sentences scored by the French XLSR ASR: base (no adapter) CER 0.008 / **WER 0.051** vs adapter CER 0.225 / **WER 0.633**. The adapter learned strong mfe/Kreol phonetics that interfere with French (expected trade-off for adapting a closely-related pair; speech is "French with a Kreol accent", ~22.5% char error). Mitigations if French preservation matters: mix French text into training (e.g. 5–10% with `[fr]`), fewer epochs (regression grows with epochs — checkpoints from epoch 2–3 would regress less), lower LR, or larger rank. mms-1b-all can't score French (no `fr` adapter) — that's why the XLSR model was used.

---

## Open questions
- Bible text license terms — not yet confirmed.
- Whether to pursue non-liturgical Morisyen audio for genuine emotion-range coverage, or accept a limited emotional range as a known constraint of this data source for now.
