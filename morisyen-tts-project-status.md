# Mauritian Creole (Morisyen) TTS + Voice Pipeline — Project Status

## ⏸️ On hold — Bible-based TTS fine-tuning (not being worked on right now)

The original plan — fine-tuning F5-TTS on Mauritian Creole using Bible audio as training data — is **paused, not cancelled**. Setting it aside for now to focus on the Gemma pipeline below. Everything already figured out is preserved here so it can be picked back up without re-deriving it.

<details>
<summary>Click to expand: paused Bible fine-tuning plan</summary>

### Goal (paused)
Fine-tune F5-TTS on Mauritian Creole (`mfe`) using Bible audio as the training source, following the same data-sourcing approach Meta used to build `facebook/mms-tts-mfe`.

### Decided so far
1. **Landscape research** — `facebook/mms-tts-mfe` (VITS) is the only dedicated public Morisyen TTS checkpoint; no larger/better-trained alternative exists, and no community LoRA/extension ecosystem exists for this language.
2. **Base model** — F5-TTS (flow-matching, DiT-based), vanilla base checkpoint, no related-language warm-start (dropped for time).
3. **License** — F5-TTS's pretrained checkpoint is CC-BY-NC; MMS-TTS checkpoints (`mms-tts-mfe`, `mms-tts-crs`, `mms-tts-fra`) are CC-BY-NC-4.0 too. Any fine-tune stays non-commercial until resolved separately.
4. **Audio** — Mauritian Creole Bible audio obtained (Faith Comes By Hearing recording, NTKM2009 translation).
5. **Alignment plan** — use `facebook/mms-fa` to align chapter audio against verse-level text.

### Remaining steps, when resumed
1. Get matching NTKM2009 text — via **find.bible** (`dev.find.bible/bibles/MFEBSM`), **bible.com/versions/344-ntkm-nouvo-testaman-dan-kreol-morisien**, or the **Digital Bible Library**. ⚠️ Text is © 2009 Bible Society of Mauritius — check DBL's stated license terms (or contact them directly) before using it at scale, separately from whatever terms the audio came under.
2. Forced alignment (`mms-fa` / `ctc-forced-aligner`) → verse-level timestamps.
3. Filter bad alignments, build the F5-TTS dataset (`wavs/` + `metadata.csv`), preprocess (`prepare_csv_wavs.py`), fine-tune (`finetune_cli.py`), evaluate against `mms-1b-all` for WER/CER.
4. Resolve the commercial-license question before any production use.

</details>

## 🎯 Active task: Gemma (audio understanding) + fine-tuned Morisyen TTS pipeline

**Gemma is audio-in, text-out only** — it doesn't generate audio itself. Its audio encoder (conformer-based, built on Google's Universal Speech Model) feeds audio in as input tokens to the LLM, unifying ASR + reasoning in one model — but speech generation still requires a separate TTS model. The two components aren't redundant; each does half the job.

### ⚠️ Dependency note
The TTS leg of this pipeline was meant to be the F5-TTS/MMS model from the paused work above. Since that's on hold, this pipeline can be built and tested end-to-end using **`facebook/mms-tts-mfe`** as-is (the existing pretrained Morisyen checkpoint, unmodified) as a placeholder output voice — swap in the fine-tuned version later without changing anything else in the pipeline.

### ⚠️ Critical unknown to test first
Gemma is trained on 140+ spoken languages, but there's no confirmation Mauritian Creole (~1.3M speakers) is among them. **Test Gemma's transcription accuracy on real Morisyen audio before building the pipeline around it.**
- **Fallback if Gemma's ASR underperforms**: use `facebook/mms-1b-all` for the ASR leg specifically (explicitly covers `mfe`), and let Gemma handle only the LLM-reasoning step on top of MMS's transcript — a two-model front end instead of one.

### Other constraints
- **Confirmed: 30-second limit per encoder pass.** Audio longer than that needs chunking before it reaches Gemma.
- **License note**: Gemma is licensed for responsible commercial use — unlike the TTS side. Gemma is not the commercial-use blocker here; the TTS checkpoint (CC-BY-NC either way, pretrained or fine-tuned) remains the actual constraint.

### Chunking approach (needed given the 30s limit)
Naive fixed-length chunking (slicing every 30s) risks cutting mid-word or mid-sentence, degrading transcription right at chunk boundaries.
- Prefer **silence-based chunking**: split on pauses (`librosa.effects.split` or a VAD tool like `webrtcvad`/`silero-vad`), keeping chunks under ~28s to leave headroom.
- If a single utterance genuinely exceeds 30s with no natural pause, fall back to a hard split at the nearest low-energy point.
- For multi-turn conversation audio, chunk per speaker turn where possible — sets up cleanly for feeding Gemma one turn at a time in the reasoning step.

### To do
- [ ] Test Gemma's ASR on sample Morisyen audio clips — confirm whether it can transcribe at all before relying on it.
- [ ] If Gemma's Morisyen ASR is weak/absent: wire `mms-1b-all` as the ASR leg instead, feeding its transcript into Gemma for the reasoning/response-generation step.
- [ ] Decide the response-generation language: Gemma generates replies directly in Kreol Morisien text, or generates in a well-resourced language and translates — depends on how strong Gemma's Morisyen generation turns out to be.
- [ ] Wire up `facebook/mms-tts-mfe` as the placeholder output voice (see dependency note).
- [ ] Build the glue code: audio in → (Gemma or MMS ASR) → transcript → Gemma reasoning → response text → TTS → audio out.
- [ ] Implement silence/VAD-based audio chunking for inputs over 30s.
- [ ] End-to-end latency check once both legs are wired.

---

## Open questions
- Whether "opencode" use here means an internal research build, or something intended for broader release — matters most once the paused Bible-text licensing question is picked back up.