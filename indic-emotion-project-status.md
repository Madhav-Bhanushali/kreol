# Indic Emotion TTS — Project Status (Marathi, female voice)

Fully independent project — **separate from the Morisyen/Kreol work** (do not touch/read/write its
directories or artifacts). Shares only: the base Chatterbox checkpoint + the
`gokhaneraslan/chatterbox-finetuning` toolkit (runtime venv + base checkpoint read-only).

**Scope (per owner):** fine-tune Chatterbox for **Marathi** as a **single-language, single-speaker
(female)** emotion-controllable voice. Phase 0 (below) is the decision gate: verify the emotion
knobs produce real variation before any training.

---

## 🎯 Voice requirement (MANDATORY — read first)

- **All Marathi generations in Phase 0 and any future Marathi work MUST use the FEMALE Marathi speaker** (`mr_female`) — never default to whichever speaker loads first.
- Confirmed from `reenigne314/chatterbox-indic-lora` `conds/conds_manifest.json`:
  - `mr_female.pt` → lang `mr`, **gender: female** (reference wav `data/rasa/mr/wavs/mr_female_017192.wav`)
  - `mr_male.pt` → lang `mr`, gender: male (DO NOT use)
  - Loader mechanism: fork's `from_indic_lora(speaker=...)` / `Conditionals.load(conds/<speaker>.pt)`.
- Hindi sanity baseline uses `hi_female` (no explicit requirement, kept consistent).
- Phase 1 training data must be **female-speaker-only** Marathi rows; if gender labels are missing,
  cross-reference `ai4bharat/indic-parler-tts` speaker metadata. If female-only data is thin, **stop
  and report actual hours / speaker count** — do not silently train on too little or fall back to male.

---

## 🧪 Phase 0 — Emotion-knob variation test (decision gate; NO training yet)

- [x] Fresh project folder `/root/bonsai/indic-emotion/` (fork source + base V2 + indic-lora assets) — nothing in the Morisyen project touched.
- [x] Confirmed female Marathi speaker (`mr_female.pt`, manifest gender=female).
- [x] **Base-version finding:** `reenigne314/chatterbox-indic-lora` is trained on Chatterbox **V2** (`t3_mtl23ls_v2`), NOT the shared V3. Phase 0 therefore uses **V2 base for BOTH Hindi (baseline) and Marathi (adapter)** so the comparison is on the same base. Flagged to owner; V3-Hindi baseline can be added on request.
- [x] Hindi (base, `hi_female`): test sentence at exaggeration 0.3 / 0.5 / 0.8 / 1.2 → 4 wavs.
- [x] Marathi (indic-lora, `mr_female`): test sentence at exaggeration 0.3 / 0.5 / 0.8 / 1.2 → 4 wavs.
- [x] Sweep `cfg_weight` and `temperature` independently at fixed exaggeration — Hindi (base) + Marathi (indic-lora female).
- [x] All outputs saved clearly labeled by language/model/parameter value in `indic-emotion/phase0_wavs/` (20 wavs; also fetched locally to `Desktop\int\indic_phase0_samples\` for listening).
- [ ] **Report to owner for listening — decision point: do NOT auto-proceed to Phase 1.** (Does the variation look real variation or flat output?)

---

## 🎯 Phase 1 — Data research + fine-tune plan (NOT started; blocked on Phase 0 decision)

- [x] **Data source — RESOLVED (research 2026-08-20):** plan's assumptions were wrong on two counts.
     1. **`ai4bharat/Rasa` DOES cover Marathi** — 22 language configs (incl. a full `Marathi` config: 26,960 train / 2,995 test rows, ~17 GB train audio; 48 kHz mono; per-utterance `style` and `gender` fields; 6-Ekman expressive + neutral + commands/conversations/news/narration; cc-by-4.0; gated — must accept terms). The indic-lora manifest's `data/rasa/mr/` reference points here — i.e. the indic-lora Marathi adapter was trained on Rasa Marathi data.
     2. **No dataset literally named "Rasmalai" exists on HF** under any ai4bharat author (API searches: `search=rasmalai`, author ai4bharat/ai4bharat-indic-speech/ai4bharat-indic-scribers → all empty). Rasmalai (Interspeech 2025, arXiv:2505.18609) is a **paper + annotation pipeline**: 13,000 h / 23 Indic langs + English / 24 M text-description annotations built ON TOP of existing corpora (Rasa, IndicVoices-R, etc.) for the **IndicParlerTTS** family — not a standalone downloadable Marathi corpus.
     3. The "~400 h / 13 languages / 6 Ekman" figure is a conflation: the **6-Ekman expressive subset originally covered only 3 languages (as/bn/ta)** per ai4bharat.iitm.ac.in; the 13-language figure comes from Indic-TTS; IndicParlerTTS used a 288 h / 9-language Rasa slice (Marathi therein = 122.47 h / 54,894 utterances, incl. non-emotional styles). Per-language emotion-row counts in the Marathi Rasa config are NOT yet verified — the config's `style` field is the way to measure it.
- [x] **Decision (owner, 2026-08-20):** Phase 1 source = **`ai4bharat/Rasa` Marathi config directly** (option A). Gated terms accepted (account Fealtyy). Downloaded ~17 GB to box `/root/bonsai/indic-emotion/rasa_marathi/` (57 train + 7 test parquet, 64/64, audio embedded). Verified counts below.
- [x] **Marathi counts (verified by scanning local parquet, 2026-08-20):**
  - **Speakers: exactly 2 — `MAR_F` (female) and `MAR_M` (male).** Single female speaker → single-speaker female training is well-defined.
  - train: 26,960 rows / **49.9 h**; test: 2,995 / 5.5 h. Dur 0.39–43.8 s, median ~5.9 s.
  - **Female (`MAR_F`): 13,924 rows / 25.95 h (train)**, test 1,548 / 2.85 h.
  - **6 Ekman emotions present** (ANGER, DISGUST, FEAR, HAPPY, SAD, SURPRISE): all = 5,221 rows / 12.01 h train (1.34 h test); **female = 2,631 rows / 6.08 h train** (0.68 h test), ~0.95–1.1 h per emotion (balanced).
  - Neutral/other female: WIKI 8.27 h + CONV 2.94 h + BOOK 2.74 h + NEWS 1.98 h + PROPER NOUN 1.76 h etc.
  - Style labels: ALEXA, ANGER, BB, BOOK, CONV, DIGI, DISGUST, FEAR, HAPPY, INDIC, NEWS, PROPER NOUN, SAD, SURPRISE, UMANG, WIKI. `gender` + `style` fields native.
  - Comparison: Morisyen had 9,645 clips / 39.8 h; here female Marathi alone = 13,924 clips / 25.95 h with 6.08 h emotion-labeled — comparable scale.
- [ ] **Training go/no-go (owner):** numbers above look sufficient (25.95 h female, balanced 6 emotions). If greenlit: train LoRA r=32 q/k/v/o on V2 base for `mr` female, warm-start from Hindi `hi` row, lock speaker embedding (single-speaker), data = female-only Rasa Marathi (all styles, emotion-weighted or all-in), reserve test split for eval.
- [ ] Filter to **female-speaker-only** rows (Rasa has a `gender` field; cross-reference `ai4bharat/indic-parler-tts` metadata if needed).
- [ ] Warm-start new `mr` embedding row from **Hindi (`hi`)** row (not `fr`).
- [ ] Decide whether to **lock/fix the speaker embedding** during training (single-language single-speaker) — document reasoning either way.
- [ ] LoRA r=32, targets q/k/v/o, `gokhaneraslan/chatterbox-finetuning` toolkit — unchanged.

---

## 🔍 Research notes (verified 2026-08-20)

- **`ai4bharat/Rasa` = the real Marathi source.** 22 configs incl. `Marathi` (train 26,960 / test 2,995, ~17 GB). Features: `filename, text, language, gender, style, duration, wav_path, audio`. gated ("auto" — accept terms), cc-by-4.0, 388 GB total, 44 speaker-language pairs / 22 languages, 48 kHz mono. The indic-lora manifest's `data/rasa/mr/wavs/mr_female_017192.wav` belongs to this dataset — resolved.
- **Rasmalai is NOT a distinct HF dataset.** It is a paper (Interspeech 2025, arXiv:2505.18609) + annotation pipeline over existing corpora (Rasa, IndicVoices-R, ...) → 13,000 h / 24 languages / 24 M text descriptions, used for IndicParlerTTS. No "rasmalai" dataset on HF (verified via API: search=rasmalai, author ai4bharat*, all empty).
- **6-Ekman expressive subset originally = only as/bn/ta** (ai4bharat.iitm.ac.in); the "13 languages / ~400 h" memory conflates Indic-TTS's 13 langs and IndicParlerTTS's 288 h / 9-lang Rasa slice (Marathi slice there = 122.47 h / 54,894 utt, mixed styles). Marathi per-emotion row counts unverified — measure via the `style` field once gated access is accepted.
- Indic-lora claims Malayalam CER 0.86 (experimental) — irrelevant here (Marathi-only).
- **Marathi coverage/hours/gender are now resolved** (full counts above). Female-speaker data: 25.95 h train / 2.85 h test; 6-Ekman female: 6.08 h / 0.68 h. No thin-data concern.
- Rasa Marathi parquet files are partitioned by gender (first files all male) — filenames `MAR_[FM]_<STYLE>_<NNNNN>`, wavs 48 kHz mono embedded in parquet.