# Custom build report

Everything from `IMPROVEMENT_PLAN.md`, implemented on branch
`claude/keyboard-voice-suggestions-d59gth` in both repos (`android-keyboard`, `voice-input`).

**What you get**
- **FUTO Keyboard Custom**: package `org.futo.inputmethod.latin.custom`.
- **FUTO Voice Input Custom**: package `org.futo.voiceinput.custom`.
- Both install next to the official FUTO apps without replacing them.
- They show "Custom" in their names and use the yellow dev-build icon so you can tell them apart.
- They never try to update themselves to the official apps.
- Both are built for 64-bit ARM only, which covers your S25 Ultra.

---

## 1. Install

1. Copy the two APKs to the phone and open them.
   - `FUTO-Keyboard-Custom.apk`
   - `FUTO-Voice-Input-Custom.apk`
   - Allow "install unknown apps" for your file manager if asked.
2. Open **FUTO Keyboard Custom** and follow its setup.
3. Pick it with the keyboard switcher.
   - In Android's keyboard list it appears as "FUTO Keyboard Custom", alongside your existing FUTO Keyboard.
4. The keyboard's settings and learned data are separate from the official app. It starts fresh.

Both APKs are signed with the debug key from the repo, not FUTO's key. That's why Android treats them as different apps, which is what you wanted.

---

## 2. What changed, by phase

### Phase 1: Voice input (keyboard)

| Stage | Status | What it does |
|---|---|---|
| 1.1 Text already in the box as context | Done, tested | Whisper models get the last ~300 characters before the cursor as context. This helps names, punctuation, casing and topic. Never used in password or "no learning" fields. Setting: *Voice Input → Use the text you've already typed*. |
| 1.2 Whisper large-v3-turbo | Done, tested with the real file | Import the GGUF (see §4) and it's used as the voice model. The import screen now recognises voice GGUFs, which it previously mistook for typing models. |
| 1.3 Newer engine + GPU | Done. GPU tested on your phone: **about half the energy of the CPU with Parakeet** | See below and §8. |

**Engine (1.3)**
- The keyboard's voice input now runs on **transcribe.cpp**, the engine FUTO's newer standalone Voice Input app uses, instead of the old whisper.cpp copy. The voice screen looks and works the same.
- Besides Whisper, the new engine can run **Parakeet, Nemotron, Moonshine and FUTO's asr4all** GGUF models.
- I turned on the ARM "dot-product" CPU instructions. Your phone has them, and they make the speech model and the typing model noticeably faster.

**GPU (1.3)** (worked out together with you after the first build; full story in §8)
- The keyboard's speech engine runs on the phone's GPU through **OpenCL**, using ggml's general OpenCL programs.
  - Vulkan was tried first and dropped: it returned empty text on your Adreno 830.
  - Qualcomm's specially tuned programs are switched off: one of them crashes the phone's GPU compiler.
- **Measured on your S25 Ultra with Parakeet:**
  - The GPU uses **about 50–60% less energy** per transcription than the CPU.
  - It's slightly faster: ~1.1 s versus ~1.3 s for 11 s of speech.
  - It gives identical text.
- **Only Parakeet-type models use the GPU.** Whisper models gave wrong text on it, so they always run on the CPU, whatever the setting says.
- **Compiled once, then cached.** The GPU programs are saved after the first compile, so the ~5 s first-run freeze shouldn't come back.
- **Safety nets:**
  - Empty GPU output is redone on the CPU.
  - A GPU crash makes the next run use the CPU.
  - "Last voice input run" in Voice Input settings shows what happened.
- **The OpenCL driver is your phone's own Qualcomm one.** It's loaded through a small shim library; if a phone doesn't provide it, voice input just runs on the CPU.
- One extra Khronos submodule was added (with your approval): `OpenCL-Headers` v2026.05.29, header files only.
- The standalone Voice Input app is still CPU-only.

**Voice Input app**
- Added **Whisper Turbo 809 (large-v3-turbo)** as a downloadable model. It downloads from Hugging Face and the SHA-256 is checked after download.
- Fixed the audio-window bug for stock models.
- Same CPU speed-up as the keyboard.
- The standalone app can't see your text box, so 1.1 doesn't apply to it.

**Important fix for turbo (both apps)**
- FUTO's own Whisper models are "acft" fine-tunes that work with a shortened audio window. The engine applied that shortening to every model.
- That hurts stock models like large-v3-turbo, so they now get the full 30-second window.
- FUTO's acft models (and every legacy `.bin`) still use the short window.

### Phase 2: Personalisation and suggestions

**What I found:** FUTO has transformer personalisation **switched off entirely** in this code.
- The training worker was commented out.
- The setting was hidden.
- A hardcoded `trainingEnabled = false` meant your typing history was never even recorded for the AI model.

So the only learning on your phone has been the older word-history dictionary. That is very likely why personalisation feels "not optimal".

| Stage | What's implemented |
|---|---|
| 2.1 "What it learned" screen | *Settings → Text Prediction → Personalisation (what it learned)*. Shows training runs with before/after quality on your own held-out typing and on everyday English. Lists recently learned words (tap ✕ to forget one) and the Roman Urdu vocabulary. Has a strength slider, typing-model threads, **Train now**, **Undo last training** and **Reset all personalisation**. |
| 2.2 Separate adjustments | Training produces a separate adapter file (`<model>.lora`; the previous one is kept as `.lora.prev`). It's applied when the model loads, at the strength you choose. The base model file is never modified. A broken adapter falls back to the plain model. |
| 2.3 Roman Urdu | Words typed in mostly-Roman-Urdu sentences go into their own vocabulary. Autocorrect leaves them alone in Roman Urdu sentences, and in English too once you've used a word 3+ times. They're kept out of the English word-history dictionary and English training, and offered as completions only in Roman Urdu sentences. Words that are also English ("main", "to", "the", "bus", …) never count as Urdu. |
| 2.4 Rolling history | Your typing history is kept **encrypted** (AES-256-GCM key in the phone's secure hardware) for 60 days, up to 20,000 words. Each training run retrains from scratch on all of it, so learning is stable instead of drifting. Deleting a word removes it for real. |
| 2.5 Bigger runs, more context | Uses up to 160 characters of preceding text (96 tokens), not just the last few words. Steps scale with your data (32–300, about one pass; the biggest run is roughly an hour while charging). FUTO's code was silently capped at ~15 steps by an "epochs = 1" setting; that's fixed. |
| 2.6 Weighting | Words you typed or picked yourself count more than autocorrected ones. When you undo an autocorrection with backspace, the keyboard learns your original word was right. |
| 2.7 Instant learning | New words and pairs you use show up as suggestions straight away, without waiting for training. English and Roman Urdu are kept separate. These are never placed first while you're typing a word, so they can't trigger autocorrect. |
| 2.8 Threads | The typing model now uses 2 threads instead of 1. You can change it from 1 to 4 on the personalisation screen. |

**When training happens:** about every 12 hours while charging with battery not low. FUTO also required "device idle", which many phones almost never report. You can also tap **Train now**.

**Safety checks before keeping a new adapter:**
- If it makes predictions on your own held-out typing worse, it's discarded.
- If it makes **everyday English** noticeably worse, it's also discarded.
- Either way the previous adapter is kept, and the result is shown on the screen.

### Phase 3: Live grammar correction (new feature)

- After each word (space or punctuation) and a 300 ms pause, the keyboard rechecks the sentence you're typing.
- It looks at commonly confused words: their/there/they're, its/it's, your/you're, then/than, to/too, a/an, "he go → he goes", "they was → they were", "could of → could have", "I seen → I saw", "have went → have gone", and about 20 more groups.
- It uses the keyboard's existing typing model, so it needs no extra memory.
- **Confident fixes are applied automatically.** Press **backspace right after** to undo, and it won't suggest that fix again for that context.
- **Less confident fixes are underlined** using Android's grammar underline. In most normal text fields, tapping the word shows the fix.
- Mostly-Roman-Urdu sentences are skipped.
- Settings: *Text Prediction → Live grammar correction* and *Apply confident fixes automatically*.
- **Stage 3.2 (small dedicated grammar model): not needed.** The plan said to add it only if 3.1 missed too much. Tested on the real model, 3.1 caught **15 of 15** typical mistakes and changed **0 of 34** correct ones (details in §3).

### Not done (optional items in the plan)

- **A bigger typing model** and an **"acft" fine-tune of large-v3-turbo** both need a proper GPU training run on a big dataset. That isn't possible in this environment or on a phone, so neither was attempted.

---

## 3. Testing

I couldn't run the apps on a phone, so testing happened on this machine, mostly against the **real models**.

**Unit tests (JVM): 58 tests, all passing**
- Grammar checker logic (13 tests): fixes, confidence levels, capitalisation, sentence boundaries, last-word caution, Roman Urdu skipping, rejected fixes not coming back.
- Confusion sets (3 tests) and Roman Urdu detection (4 tests).
- Personalisation logic (18 tests): retention, weighting, long context, typo pairs, undone autocorrect, Roman Urdu exclusion, held-out split, training size, instant memory, Roman Urdu vocabulary and protection.
- GGUF reader (4 tests): typing model vs voice model detection.
- Power-test maths (7 tests): unit detection, idle subtraction, combining rounds, detecting wrong GPU transcripts.
- **Real voice engine** (9 tests): the keyboard's actual `WhisperGGML` class and engine built for this machine.
  - FUTO's built-in model transcribed the JFK "ask not what your country…" recording correctly.
  - Text context worked: the output picked up the comma from the context.
  - GPU request falls back to the CPU, a previous GPU crash forces the CPU, and an invalid model is rejected.
  - **Whisper large-v3-turbo** transcribed it correctly in 26.9 s on this machine's CPU.
  - **NVIDIA Parakeet TDT 0.6B v2** transcribed it correctly in **2.2 s**, with perfect punctuation.
  - Both files are the exact ones in §4.
  - The GPU rule reads each file's header correctly: Whisper `.bin` and Whisper GGUF go to the CPU; Parakeet is allowed on the GPU.
- **GPU program cache** (`voiceinput-shared/src/main/cpp/tests`), run on PoCL (a desktop OpenCL implementation):
  - The first build compiles and saves; the second loads from cache and computes correctly.
  - Different build options get their own entry, and a corrupted file is detected and replaced.

**Native tests against the real typing model (`native/jni/tests_host`)**
- **Sentence scoring:** correct, deterministic, and suggestions are identical before and after grammar checks.
- **Grammar, 15 deliberate mistakes:** all caught (10 auto-fixed, 5 underlined).
- **Grammar, 34 correct alternatives:** none would trigger. The closest was well below the underline threshold.
- **Personalisation training:**
  - Trained a real adapter on ~250 varied messages containing a made-up name.
  - The learned phrase went from log-probability −45.9 to −4.1 at the default 75% strength (about e^41 times more likely).
  - Everyday English got slightly *better* (−5.89 → −5.60 per word).
  - Strength 0 gives exactly the base model, and the base model file is never changed.
  - A corrupt adapter falls back safely.
- **Bugs these checks caught and fixed**
  - With tiny repetitive data the first version over-learned and hurt everyday English. That's why the "about one pass" step count and the everyday-English check exist.
  - FUTO's trainer was capped at 1 epoch (~15 steps), so it barely learned. Removed.
  - The grammar scorer treated the last word as unfinished because of this model's tokens. Fixed, which is why the margins above are reliable.
  - The release build's code shrinker (R8) had removed the training worker's `doWork()`, because WorkManager creates workers by reflection. Training would have crashed on the phone. Added a keep rule and confirmed the method is in the final APK.

**Checked in the final APKs**
- Packages are `org.futo.inputmethod.latin.custom` and `org.futo.voiceinput.custom`, labelled "FUTO Keyboard Custom" and "FUTO Voice Input Custom".
- Both are arm64 and signed with the same fixed public debug key, so future custom builds install over them.
- The new settings, the training worker, the native grammar scorer and the voice engine's native methods are all present after shrinking.

**Not tested** (needs your phone):
- The actual UI.
- Speed on the S25.
- Underline display in specific apps.
- WorkManager training while charging.

---

## 4. Models to download (optional)

The keyboard works straight away with its built-in models: FUTO's tiny English Whisper for voice and its typing model for suggestions. For much better voice accuracy, download **one** of these and import it.

### Recommended: NVIDIA Parakeet TDT 0.6B v2 (English, fast and accurate)
- File: `parakeet-tdt-0.6b-v2-Q5_K_M.gguf` (539 MB, licence CC-BY-4.0).
- Download: https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v2-gguf/resolve/main/parakeet-tdt-0.6b-v2-Q5_K_M.gguf
- SHA-256: `dfa904520b95451599683613fea47766c378c5fdf5d8cddf48151226b6eaec85`
- **About 12× faster than Whisper large-v3-turbo** in my test (2.2 s vs 26.9 s for 11 s of audio), and it got the punctuation exactly right.
- On your phone it waits only ~0.2–0.7 s after you stop talking.
- It's the model that runs on the GPU, using about half the energy of the CPU.
- Two limitations:
  - It ignores the personal-dictionary glossary.
  - It ignores the "text already typed" context; that only helps Whisper models.

### Alternative: Whisper large-v3-turbo (the model in the plan)
- File: `whisper-large-v3-turbo-Q5_K_M.gguf` (620 MB, MIT licence).
- Download: https://huggingface.co/handy-computer/whisper-large-v3-turbo-gguf/resolve/main/whisper-large-v3-turbo-Q5_K_M.gguf
- SHA-256: `06fc9923c09fb3ae878ea12c34d1be3ed18fad039ab14b4b045c7b6c40af56b2`
- Uses your glossary and the "text already typed" context, but it's **slow**: it always processes a full 30-second window, even for a short clip.
  - About 29 s for a 3-second clip on this machine; several seconds on your phone.
  - It never runs on the GPU.
- One side effect of the context: words already in the text box can be left out if you say them again. In my test, "My fellow Americans" was given as context and dropped from the transcript. Normally you dictate new text, so this doesn't matter.

### How to import into the keyboard (either model)
1. Download the file on the phone (e.g. in Chrome).
2. Open **FUTO Keyboard Custom** settings → **Languages & Models** → **Import resource file**.
3. Pick the downloaded file. The screen should say **Voice Input Model**.
4. Choose **English** and confirm.
5. Voice input in the keyboard now uses that model.
6. To go back, delete it for English in Languages & Models; the built-in model takes over.

### Voice Input Custom app
- Pick **Whisper Turbo 809** in its model settings and it downloads itself (checksum verified).
- FUTO's own **Nemotron / Parakeet** options are also in that list.

**Typing model:** nothing to download.

---

## 5. Settings worth checking after install

- **Text Prediction**
  - **Live grammar correction**: on.
  - **Apply confident fixes automatically**: on.
  - **Personalisation (what it learned)**: learning is on by default. Strength 75%, threads 2.
- **Voice Input**
  - **Use the text you've already typed**: on.
  - **Use the GPU for voice input**: off by default. **Turn it on if you use Parakeet**; it uses about half the energy. Whisper models ignore it and use the CPU.
  - **Last voice input run** (box under the settings): model, GPU/CPU, timings, any GPU problem. **Copy details** copies it.
  - **Measure voice power use (CPU vs GPU)**: the battery test (see §8).

---

## 6. Commits

All pushed to `claude/keyboard-voice-suggestions-d59gth`. No pull requests were opened.

**android-keyboard**
- `d2157f2` Add improvement plan
- `4dc5ed6` Add 'custom' build flavor that installs alongside the official keyboard
- `e4dd710` Add live grammar correction and rebuild personalisation (Phases 2 and 3)
- `7e24ebf` Move keyboard voice input to transcribe.cpp; text context; tests (Phase 1)
- `2a904fd` Host test with the real Whisper large-v3-turbo GGUF
- `7629ea9` Tune training size; save history in background; Parakeet test
- `7c5d812` Keep TrainingWorker through R8; add this report
- `ddb937f` … `425025f` Voice speed and GPU work after your first tests (see §8):
  - `e236314` keep the voice model loaded, GPU→CPU fallback, diagnostics
  - `6088a56` engine log in diagnostics
  - `ddb937f`, `cc83624` Vulkan GPU (later removed)
  - `48b66fe` switch to OpenCL
  - `1402eae` crash-safe GPU log
  - `d08c3e3` turn off the Qualcomm-tuned programs (compiler crash)
  - `4578ec4` GPU program cache and the power test
  - `425025f` Parakeet-only GPU rule; power test uses your model

**voice-input**
- `b8037a7` Add 'custom' build flavor that installs alongside the official app
- `f7870e7` Add Whisper large-v3-turbo download, fix audio window for stock models
- `c93dd0b` Sign the custom build with a fixed public debug key

## 7. How to rebuild

```
# keyboard (needs Android SDK + NDK 28.2; submodules incl. transcribe.cpp)
# The swipe models are Git LFS files: without this the APK ships 132-byte placeholders
# and swiping crashes ("Failed to initialize SwipeDecoder")
git -C java/assets/futo-swipe lfs pull
VERSION_CODE=... VERSION_NAME=... ./gradlew assembleCustomRelease
# tests (optional: VOICE_HOST_LIB_DIR=<desktop build of voiceinput-shared/src/main/cpp>)
./gradlew testCustomReleaseUnitTest
# native typing-model tests with the real model (desktop, clang)
cmake -S native/jni/tests_host -B /tmp/xlm-host -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ && cmake --build /tmp/xlm-host
/tmp/xlm-host/test_scoring java/res-large/raw/ml4_q6_k.gguf

# voice input app (NDK 30)
VERSION_CODE=... VERSION_NAME=... ./gradlew assembleCustomRelease
```

---

## 8. After the first install: voice speed and the GPU

What we found and changed together while you tested on the phone.

**Why voice input felt slow at first**
- **Whisper large-v3-turbo is inherently slow for short clips.** It always processes a 30-second window, so you get several seconds of waiting. Parakeet fixed that: a 0.2–0.7 s wait on your phone.
- **The keyboard threw the voice model away on almost every mic press.** It unloaded the model whenever Android sent a routine "free some memory" signal, for example when the keyboard was simply hidden. It now only unloads under real memory pressure.

**Getting the GPU to work**

| Attempt | Result on the S25 Ultra | What the diagnostics showed |
|---|---|---|
| Vulkan | Empty text, slow (1.75–20 s) | `decoder: … 0 tokens`; still broken with fp16 turned off |
| OpenCL with Qualcomm-tuned programs | Keyboard crashed | 139 of 141 programs compiled, then the compiler crashed on `gemv_noshuffle_q5_k_f32` |
| **OpenCL with general programs** | **Works** | Correct text; ~5 s to compile on first use, now cached |
| OpenCL with a Whisper model | Wrong text ("!....London, oh") | Fixed by allowing only Parakeet-type models on the GPU |

**Battery measurements**
- **How the test works:** *Measure voice power use* transcribes the same 11 s recording repeatedly, alternating idle, CPU and GPU periods in two rounds. It reads the phone's battery current and voltage 10× a second and subtracts idle power.
- **Comparing runs:** compare CPU and GPU within one run, not across runs. Conditions such as brightness or temperature change the absolute numbers.

| Run (Parakeet) | CPU per transcription | GPU per transcription | GPU saving |
|---|---|---|---|
| With music playing in the background | 9.5 J, 1.33 s | 4.7 J, 1.07 s | ~50% |
| Without music | 10.6 J, 1.30 s | 4.4 J, 1.09 s | ~58% |

- Every round agreed: the CPU used 9.4–10.6 J, the GPU 4.2–4.8 J. The transcripts were identical.
- **In everyday terms:** your battery holds ~70,000 J. Half an hour of actual dictation costs ~2% of the battery on the CPU and ~1% on the GPU.
- **For comparison, FUTO's built-in tiny Whisper model** (measured by mistake on the first two runs) was cheaper still on the CPU, ~1 J per transcription. On the GPU it used ~3× more and produced garbage, which is why Whisper is now CPU-only.

**Recommendation:** use **Parakeet with the GPU on**.

