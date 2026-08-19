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
- [x] Test Gemma's ASR on sample Morisyen audio clips — confirm whether it can transcribe at all before relying on it.
- [x] If Gemma's Morisyen ASR is weak/absent: wire `mms-1b-all` as the ASR leg instead, feeding its transcript into Gemma for the reasoning/response-generation step.
- [x] Decide the response-generation language: Gemma generates replies directly in Kreol Morisien text, or generates in a well-resourced language and translates — depends on how strong Gemma's Morisyen generation turns out to be.
- [x] Wire up `facebook/mms-tts-mfe` as the placeholder output voice (see dependency note).
- [x] Build the glue code: audio in → (Gemma or MMS ASR) → transcript → Gemma reasoning → response text → TTS → audio out.
- [x] Implement silence/VAD-based audio chunking for inputs over 30s.
- [x] End-to-end latency check once both legs are wired.

### Status notes (append-only)
- **Gemma ASR tested on real Morisyen (B01 Matthew 1, 25s + 75s clips from the drama audio).** Verdict: it CAN transcribe Morisyen, but with heavy errors and unstable orthography — it mixes French spellings into the output ("ci papa" / "c'est papa", "s'appelle", "Son maman") and garbles names. On the same 25s clip, `facebook/mms-1b-all` was near-perfect ("abraam ti papa izaak izaak ti papa zakob zakob ti papa zida ek so bann frer zida ti papa perez ek zera zot mama ti apel tamar perez ti papa esron"; 1 minor error) vs Gemma's error-dense French-mixed version. → **ASR leg = `mms-1b-all`; Gemma handles reasoning only** (as the doc's fallback prescribes). Pipeline defaults to `--asr mms`.
- **Gemma's Morisyen generation is strong** — given the MMS transcript, it replied in fluent Kreol Morisien ("Sa se sa bann non bann papa ek zot gran-papa ki nou konn? Ki sa ou anvi konn plis sou zot?"). → **Decided: Gemma generates replies directly in Kreol Morisien** (no translate-then-back step).
- **`.env` correction:** the pasted `GEMMA_AUDIO_BASE_URL=http://45.194.3.34:8101/v1` is dead (unreachable from both this machine and the box). The live audio endpoint is `http://43.242.226.49:8101/v1` (same host as the text endpoint; auth `GEMMA_AUDIO_API_KEY=gemma4-secret`, verified HTTP 200 on `/v1/models`). `.env` on both machines updated.
- **Chunking verified on real audio:** 75s Matthew 1 clip → 4 silence/VAD chunks (max 27.9s), each transcribed correctly. Synthetic test confirmed the >28s single-utterance hard-split at the nearest low-energy point.
- **End-to-end pipeline works on the box** (`scripts/gemma_pipeline.py --asr mms`): 25s input → ASR 9.5s + reason 0.2s + TTS 10.0s (both model loads included on first run; warm runs faster) = 19.7s total → 7.34s reply wav. ASR/TTS model weights cache on the box so repeat runs drop well below this.
- **Live demo launched (any language in → Kreol out).** New `scripts/whisper_asr.py` (faster-whisper, auto language detection, silero VAD, handles long audio) + `scripts/live_demo.py` (Gradio UI). Verified on the box with two non-Kreol inputs: French "Bonjour, comment allez-vous aujourd'hui ?" (whisper detected `fr`) and Hindi "नमस्ते! आप कैसे हैं?" (detected `hi`) — Gemma answered both in fluent Kreol Morisien ("Mwa byen, merci! Et ou menm, kouman ou ye?" / "Mo byen, mersi. Lagwa kisa to di, to la?"), each saved as a spoken reply wav. Running on the box at **http://43.242.226.61:7860** (tmux session `live`, CPU mode so it doesn't fight the training jobs; GPU busy → set `CUDA_VISIBLE_DEVICES=''`). Reached externally (HTTP 200).
- **Gemma text endpoint quirk:** `max_tokens=512` hangs the request (read timeout); `<=400` responds in ~0.2s. `scripts/gemma_reason.py` now defaults to `max_tokens=256` with a configurable timeout (`GEMMA_TIMEOUT`, default 60s) so the demo fails fast instead of hanging.

---

## Open questions
- Whether "opencode" use here means an internal research build, or something intended for broader release — matters most once the paused Bible-text licensing question is picked back up.