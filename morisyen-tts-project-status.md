# Mauritian Creole (Morisyen) TTS — Project Status

## Goal
Fine-tune F5-TTS on Mauritian Creole (`mfe`) using Bible audio as the training source, following the same data-sourcing approach Meta used to build `facebook/mms-tts-mfe`.

---

## ✅ Done so far

1. **Landscape research** — evaluated existing options for Morisyen TTS. Conclusion: `facebook/mms-tts-mfe` (VITS architecture) is the only dedicated public checkpoint; no larger/better-trained public alternative exists, and no community LoRA/extension ecosystem exists for this language (unlike Indic languages).
2. **Base model decision** — chose to fine-tune **F5-TTS** (flow-matching, DiT-based, non-autoregressive) instead of extending MMS-TTS directly, to get F5's stronger zero-shot/voice-cloning architecture.
3. **License checked** — F5-TTS's code is MIT, but the **pretrained checkpoint is CC-BY-NC** (inherited from the Emilia training set). Any checkpoint fine-tuned from it stays non-commercial. ⚠️ This blocks production/commercial use as-is — treat this phase as a research/feasibility build, not a shippable model, until this is resolved separately.
4. **Audio sourced** — obtained Mauritian Creole Bible audio (Faith Comes By Hearing recording, the same source family Meta used for MMS).
5. **Alignment/tooling plan set** — decided to use `facebook/mms-fa` (Meta's forced-alignment model, same language coverage as MMS-TTS) to align long-form chapter audio against verse-level text.

---

## 🔲 To do

1. **Get matching text** — see below. Needed before alignment can run.
2. **Forced alignment** — run `mms-fa` (via the `ctc-forced-aligner` package) on chapter audio + verse text → per-verse timestamps.
3. **Filter bad alignments** — drop low-confidence/mismatched segments (cross-validation filtering step, same as MMS's own pipeline).
4. **Build F5-TTS dataset** — `wavs/` folder + `metadata.csv` (`audio_file|text` format), resampled to 24kHz mono.
5. **Preprocess** — `python prepare_csv_wavs.py <data_dir> <output_dir>` → generates `raw.arrow`, `duration.json`, `vocab.txt`. Manually verify `vocab.txt` covers all Morisyen characters (check accented vowels specifically).
6. **Fine-tune** — `accelerate launch finetune_cli.py` from `F5TTS_v1_Base`, `--tokenizer char`, starting `--learning_rate 1e-5`. Expect this to take real iteration — don't expect a working model from one run.
7. **Evaluate** — generate test sentences, transcribe output with `facebook/mms-1b-all` (MMS's ASR checkpoint, also covers `mfe`), compute word/character error rate against input text.
8. **Resolve the license question** — decide whether this stays a research artifact, or whether a commercially-clean path (different base model, or from-scratch training on rights-cleared data) is needed before any production use.

---

## Getting the Kreol Morisien Bible text

The audio you have is almost certainly **Faith Comes By Hearing's recording of the NTKM2009 translation** (*Nouvo Testaman dan Kreol Morisien*, published by the Bible Society of Mauritius, 2009) — this is the same translation used for `mms-tts-mfe`'s training data, and it's the only widely-available Morisyen Bible audio. Getting the matching text from the **same translation** matters — using a different Morisyen Bible edition's text against this audio will cause verse-level wording mismatches during alignment.

### Where to get it
- **find.bible** listing for this exact edition: https://dev.find.bible/bibles/MFEBSM/ — links out to the Digital Bible Library (DBL) entry and YouVersion.
- **YouVersion / Bible.com** (`NTKM2009`): https://www.bible.com/versions/344-ntkm-nouvo-testaman-dan-kreol-morisien — readable verse-by-verse in browser; has an API for developers, but check their terms before bulk-scraping.
- **Digital Bible Library (DBL)** — the canonical source for text + explicit licensing terms per translation: https://www.digitalbiblelibrary.org — search "Kreol Morisien" / "NTKM2009". This is the best place to confirm redistribution rights before using the text in a training pipeline.
- **dokumen.pub** has a readable copy of the full NTKM2009 text (useful for spot-checking verse wording, not recommended as your bulk-download source).

### ⚠️ License — check this before building anything on it
The text is **© 2009 Bible Society of Mauritius**, not public domain and not obviously open-licensed. This is separate from the audio's license, and the two don't necessarily carry the same terms. Before using this text at scale (even for a non-commercial research fine-tune):
1. Check the DBL entry for this translation's explicit license terms (DBL requires publishers to declare usage rights — CC-BY, CC-BY-NC, "digital use only," etc.).
2. If unclear, contact the Bible Society of Mauritius directly (referenced on the bible.com listing) — for a research/non-commercial ML use case, permission is often straightforward to obtain, but shouldn't be assumed.

### Practical extraction approach
Once license terms are confirmed, the actual pull is straightforward since the text is verse-tagged: fetch each book/chapter, split into verses, and store as `book_chapter_verse → text` — this verse-level structure is exactly the segmentation granularity you want for aligning against the chaptered audio (see "To do," step 2).

```python
# illustrative structure only — actual fetch depends on which source above you're granted access through
verses = {
    "MAT.1.1": "Lalis zanset Zezi-Kri, fis David, fis Abraam...",
    "MAT.1.2": "Abraam ti papa Izaak...",
    # ...
}
```

---

## Preprocessing (detail on "To do" steps 2–4)

This is the stage between "raw chapter audio + verse text" and "clean dataset ready for `prepare_csv_wavs.py`." Skipping or rushing this is the most common reason a fine-tune produces garbled or mispronounced output later — worth treating as its own careful pass, not a quick script.

### A. Audio preprocessing

1. **Resample to a consistent rate.** F5-TTS expects 24kHz mono. Bible audio distributed via Faith Comes By Hearing is often 44.1kHz or 48kHz stereo — convert everything up front so every later step works on identical formats.
   ```python
   import torchaudio

   wav, sr = torchaudio.load("chapter_audio.wav")
   if wav.shape[0] > 1:                      # stereo → mono
       wav = wav.mean(dim=0, keepdim=True)
   wav = torchaudio.functional.resample(wav, sr, 24000)
   torchaudio.save("chapter_audio_24k.wav", wav, 24000)
   ```

2. **Trim leading/trailing silence per segment**, after alignment splits the chapter into verse-level clips. Silence at clip boundaries confuses duration modeling during training.
   ```python
   import librosa, soundfile as sf

   y, sr = librosa.load("seg_0001.wav", sr=24000)
   y_trimmed, _ = librosa.effects.trim(y, top_db=30)
   sf.write("seg_0001_trimmed.wav", y_trimmed, sr)
   ```

3. **Normalize loudness** across all clips so the model isn't learning volume as a spurious feature. Peak or RMS normalization to a consistent target (e.g. -20 dBFS) is standard:
   ```python
   from pydub import AudioSegment

   audio = AudioSegment.from_wav("seg_0001_trimmed.wav")
   change = -20.0 - audio.dBFS
   audio.apply_gain(change).export("seg_0001_norm.wav", format="wav")
   ```

4. **Check for recording artifacts.** Older/field-recorded Bible audio sometimes has tape hiss, room reverb, or mic clipping. A quick pass with a noise-reduction tool (e.g. `noisereduce` library, spectral-gating based) can help — but don't over-clean; artifacts introduced by aggressive denoising (metallic/robotic residue) are arguably worse for TTS training than mild original noise, since the model will faithfully learn whatever's in the training audio, denoising artifacts included.

5. **Filter by duration.** Drop segments outside roughly 1–15 seconds — too short gives the model too little context per example, too long causes memory/alignment issues during training. Verse-level segmentation from Step 2 (forced alignment) should naturally land most clips in this range, but check the distribution and drop outliers rather than truncating them, since a truncated clip's audio and text will no longer match.

### B. Text preprocessing / normalization

1. **Strip verse numbers and formatting artifacts** — Bible text sources often embed verse numbers, footnote markers, or cross-reference symbols inline (e.g. `29Si Bondie...`). These must not end up in the training text, since they're not read aloud in the audio and would corrupt the text-audio pairing.
   ```python
   import re

   def clean_verse(text):
       text = re.sub(r"^\d+", "", text)          # leading verse number
       text = re.sub(r"\[.*?\]|\(.*?\)", "", text) # footnote/cross-ref markers, if present
       return text.strip()
   ```

2. **Normalize orthography consistency.** Mauritian Creole spelling has some variation between the older *Morisyen* spelling and the post-2011 standardized *Kreol Morisien* orthography (silent letters removed, etc. — see the earlier note on the language's orthographic reform). If your source text mixes conventions across chapters/books, standardize to one system before training — the tokenizer treats spelling variants as entirely different tokens, so inconsistency here directly fragments your vocabulary and dilutes training signal for what should be the same word.

3. **Normalize punctuation and casing** consistent with how F5-TTS's `char` tokenizer will treat them — decide up front whether to keep sentence-initial capitalization and terminal punctuation (generally yes, it carries prosodic information — question marks and periods correlate with real pitch/pause differences) or lowercase everything (only do this if you're seeing the model over-fit to capitalization patterns rather than genuine prosody, which is less likely to be the issue here).

4. **Cross-check every verse's text against its aligned audio duration** as a sanity filter — if a verse's character count implies a wildly different speaking duration than what the aligner found (e.g. very short audio for a very long verse), that's very likely a bad alignment, not a genuine fast-talker — drop it rather than trust it.

### C. Train/validation split

Hold out a small slice (5–10%) of verses — ideally spanning multiple books/chapters rather than one contiguous block, so the held-out set isn't systematically easier or harder — to track whether the model is actually generalizing or just memorizing during fine-tuning. This is what you'll use for the WER/CER check against `mms-1b-all` in the evaluation step, rather than eyeballing training-set samples (which will look deceptively good since the model has seen them).

---

## Open questions to resolve before continuing
- Which exact source you'll use for text (DBL API vs. YouVersion vs. direct contact with Bible Society of Mauritius) — depends on what license terms come back.
- Whether "opencode" use here means an internal research build, or something intended for broader release — this changes how carefully the text license needs to be nailed down before proceeding.
