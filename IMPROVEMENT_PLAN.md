# Keyboard improvement plan

Planning document for improving voice input, typing suggestions/autocorrect,
personalisation, and adding live grammar correction.

## Ground rules

- **Local only.** Every model runs on the device. Nothing is sent off the phone.
- **Target device:** Samsung Galaxy S25 Ultra. Larger models than FUTO ships by
  default are fine, but the keyboard must stay light on memory, since Android
  kills keyboards that use too much.
- **Languages:**
  - Typing is mainly English.
  - Some typing is Roman Urdu (Urdu written in English letters). Those sentences
    are mostly Urdu with a few English words mixed in, never the other way round.
  - Voice input is English only.
- **Scope:** changes go into the keyboard app only (including its built-in
  `voiceinput-shared` module). The standalone `voice-input` app is not changed.
- **Testing:** each phase is tested on the phone by the user; code is written
  and committed on branch `claude/keyboard-voice-suggestions-d59gth`.

Recommended order: Phase 1 → Phase 2 → Phase 3. The phases are independent and
can be reordered.

---

## Phase 1: Voice input

### Current state
- The best available model is Whisper "small" (English-only or multilingual),
  8-bit (`voiceinput-shared/.../Models.kt`).
- These are FUTO's "acft" models, fine-tuned to work when the audio window
  (`audio_ctx`) is shortened to the clip length
  (`native/jni/org_futo_voiceinput_WhisperGGML.cpp:101`).
- The only prompt given to Whisper is the glossary
  (`voiceinput-shared/.../whisper/MultiModelRunner.kt:63`).
- The bundled whisper.cpp is old and CPU-only.

### Stage 1.1: Give the speech model the text already in the box
- Pass the text before the cursor (trimmed to the last few sentences) as the
  Whisper prompt, together with the existing glossary.
- Benefit: better names, capitalisation, punctuation style and topic continuity.
- Size: small, low risk.

### Stage 1.2: Add Whisper large-v3-turbo as a model option
- Add it as a downloadable/importable model alongside the existing ones.
- Run it with the full 30-second audio window (`audio_ctx = 1500`) at first,
  because stock models lose accuracy when the window is shortened. Short clips
  will be slower than today's models until Stage 1.3 is done.
- Check that the bundled whisper.cpp loads it (it uses 128 mel bins instead of
  80; the loader appears to read this from the file, but this is untested).
- Optional later: fine-tune turbo the "acft" way so the shortened window works.
- Size: small to medium.

### Stage 1.3: Update the speech engine (whisper.cpp) and use the GPU
- Update the bundled whisper.cpp to a current version.
- Carry over FUTO's modifications (allowed/forbidden language lists, "bail"
  languages, and anything else they changed).
- Enable GPU acceleration (Vulkan or OpenCL on the Adreno GPU) and flash
  attention, falling back to the CPU if the GPU path fails.
- Size: medium to large. This is the step that makes large-v3-turbo fast.

### Not doing
- A better "finished speaking" detector (e.g. Silero VAD). Voice input is
  always stopped manually, so it isn't needed.

---

## Phase 2: Personalisation and suggestions

### Problems found in the current code
1. **Learns rarely.** Training only runs while charging, idle and with enough
   battery, at most every 20 hours, and only after 100+ logged words
   (`java/src/.../xlm/TrainingWorker.kt:261-278`, `:67`).
2. **Small training runs.** A fixed 128 training steps on 64-token chunks
   (`native/jni/org_futo_inputmethod_latin_xlm_AdapterTrainer.cpp:71-77`), so
   much of the logged data is never used.
3. **Changes are merged permanently.** The trained adjustment is merged into
   the model file and overwrites it; the backup is a commented-out TODO
   (`TrainingWorker.kt:198-206`). The model can drift over time with no undo.
4. **Data is deleted after each run.** Each night's batch is seen once, so the
   model swings toward recent text and forgets older patterns.
5. **Short context.** Only the few previous words (`ngramContext`) are used;
   the longer `priorContext` is recorded but never used.
6. **Shallow unlearning.** Deleting a suggestion only removes it from the
   not-yet-trained log; anything already merged stays.
7. **Silent skips.** Words are not learned when the detected language doesn't
   match the model, or when context extraction fails
   (`java/src/.../xlm/LanguageModelFacilitator.kt:529-546`).
8. **Autocorrected words count as chosen.** Words inserted by autocorrect are
   logged like words the user picked, so autocorrect mistakes can reinforce
   themselves.
9. **Roman Urdu is probably treated as English typos**, which causes unwanted
   autocorrections and pollutes English predictions.

### Stage 2.1 (F): "What it learned" screen
- Show recent learned items, training run history, and before/after quality.
- Controls to delete individual items and to reset all personalisation.
- Done first, so the real problems can be seen and later fixes can be checked.

### Stage 2.2 (A): Keep adjustments separate from the base model
- Save the trained adjustment (LoRA adapter) as its own file instead of merging
  it into the model.
- Load it on top of the base model at runtime.
- Enables reset, roll back to the previous adapter, and a strength setting.

### Stage 2.3: Roman Urdu as its own vocabulary
- Detect sentences that are mostly Roman Urdu.
- Store Roman Urdu words as a separate learned vocabulary.
- Autocorrect stops "fixing" words like "acha", "nahi", "kya".
- Roman Urdu words don't appear in English predictions, and English words used
  inside Roman Urdu sentences still work.

### Stage 2.4 (B): Keep a rolling typing history and retrain on all of it
- Keep an encrypted rolling window of typing history (e.g. the last 30–60 days)
  instead of deleting it after each run.
- Retrain the adapter on the whole window each time, giving stable learning
  instead of drift.
- Deleting an item removes it from the history; the next retrain forgets it
  properly.

### Stage 2.5 (C): Bigger nightly training runs with more context
- Scale the number of training steps to the amount of data.
- Train on longer context, using the recorded `priorContext`, so it learns
  phrasing, not just the last few words.

### Stage 2.6 (E): Weight words by how they were chosen
- Words typed or tapped by hand count more than words autocorrect inserted.
- Autocorrections that were undone count as negative examples.

### Stage 2.7 (D): Learn new words immediately
- A lightweight on-device memory of recent words and phrases that boosts
  suggestions straight away, so new words and names work the same day instead
  of after overnight training.

### Stage 2.8: Run the typing model on more cores
- Raise the typing model from 1 thread to 2–4
  (`native/jni/src/ggml/LanguageModel.cpp:59-60`), measuring latency so typing
  never lags.

---

## Phase 3: Live grammar correction (new feature)

The keyboard currently only corrects the word being typed. Words earlier in
the sentence never change, so correctly spelled but wrong words ("I seen there
car, its a nice one") are never fixed. This phase adds sentence-level checking
while typing.

### Behaviour (decided)
- Runs automatically as you type, after each completed word or punctuation
  mark, with a short pause (about 300 ms). Typing cancels and restarts the check.
- **High-confidence fixes are applied automatically**, briefly highlighted, and
  backspace undoes them.
- **Lower-confidence fixes are underlined**; tapping the word accepts the fix.
- Sentences that are mostly Roman Urdu are skipped entirely.

### Stage 3.1: Commonly-confused words, using the existing typing model
- No extra model and no extra memory.
- For each word in the current sentence, check common confusion sets, e.g.:
  - their / there / they're
  - its / it's
  - your / you're
  - a / an
  - then / than
  - to / too / two
  - verb agreement ("he go" → "he goes")
- Score each alternative version of the sentence with the existing typing
  model; switch or underline only when the alternative is clearly more likely.
- Confidence thresholds decide auto-apply vs underline.

### Stage 3.2 (only if 3.1 misses too much): Small dedicated grammar model
- A compact model (about 150–250 MB) built specifically to tag which words need
  fixing, not a general chatbot model, so it is smaller, faster and rewrites
  wording less often.
- Loaded while the keyboard is open, unloaded when it closes.
- Runs in a separate process so a crash or memory pressure cannot take the
  keyboard down.
- Same auto-apply/underline rules and Roman Urdu skipping as Stage 3.1.
- Larger general LLMs (~1 GB) were ruled out as too heavy for a keyboard.

---

## Optional / later
- **Bigger typing model:** train a larger typing language model (about
  100–300M parameters) in FUTO's format (key-blending setup and metadata, see
  `native/jni/src/ggml/ModelMeta.cpp`). The biggest improvement for
  suggestions, but it needs a real training run.
- **"acft" fine-tune of large-v3-turbo** so it can use the shortened audio
  window (see Stage 1.2).

## Notes
- FUTO's licence (FUTO Source First 1.1) allows modification for personal use.
- FUTO does not accept AI-generated pull requests, so this work is for a
  personal build, not for sending upstream.
