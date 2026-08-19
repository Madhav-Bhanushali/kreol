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
- [ ] Load base Chatterbox V3 checkpoint (`t3_model="v3"`), inspect tokenizer vocab for any missing Morisyen-specific characters.
- [ ] Add `mfe` language embedding row, initialized as a copy of the `fr` row (not random).
- [ ] Configure LoRA (rank-32 starting point, target `q/k/v/o` in T3, freeze S3Gen + speaker encoder + all other language rows).
- [ ] Run fine-tune via `gokhaneraslan/chatterbox-finetuning`.
- [ ] Evaluate: WER/CER against `facebook/mms-1b-all` transcription of generated audio; listen-test at a couple of `exaggeration` settings to see whether emotion control transfers at all to `mfe`, even if range is limited by the formal-register training data.
- [ ] Regression check: confirm French (and a couple of other original 23 languages) still generate correctly after the `mfe` fine-tune — LoRA + frozen embeddings should protect this, but verify rather than assume.

### Status notes (append-only)
- **Data-source correction:** `mfe_mu` is **not** National Assembly proceedings — every one of the 10,237 rows has `source = jw_bible_kreol_morisien` (the JW Bible in Kreol Morisien, copyright-restricted translation). User confirmed to proceed with it anyway. The license question is therefore still open, same shape as the old Bible plan; attribution/non-commercial constraints likely apply.
- **Dataset inspection (md#1):** 10,237 train rows / 42.1h total audio / all 24 kHz / clips ~15s (constant; min 3.6s). CER: min 0.0, p10 0.095, median 0.191, p90 0.282, max 0.498, no NaN. Test split: 539 rows. Actual quality columns are `snr` + `dnsmos_sig/bak/ovr/p808` (not `wada_snr`/`dnsmos_p835` as the md listed).
- **CER filter (md#2):** chose **CER < 0.3** → keeps 9,681 rows / 39.8h (95%); CER < 0.2 would keep 22.5h, CER < 0.25 keeps 32.0h. Re-runnable via `--cer`.
- **LJSpeech build (md#3):** `scripts/build_ljspeech_dataset.py` → `data/worldspeech_mfe_ljspeech/` (`wavs/` 24 kHz mono PCM-16 + `metadata.csv` as `id|human_transcript|human_transcript`). Workaround: `datasets` 5.0.1 decodes audio via `torchcodec`, whose binary mismatches the box's torch 2.8 (undefined symbol) — the script loads the audio column as raw bytes and decodes with `soundfile` instead. HF token now in `.env` (gitignored) on both machines.
- **Chatterbox is a new toolchain** — not yet installed anywhere; the box has space now (198G free after the user's cleanup), so model/dataset downloads go to the box.

---

## Open questions
- Bible text license terms — not yet confirmed.
- Whether to pursue non-liturgical Morisyen audio for genuine emotion-range coverage, or accept a limited emotional range as a known constraint of this data source for now.
