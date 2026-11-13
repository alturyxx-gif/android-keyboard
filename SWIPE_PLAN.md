# Swipe Accuracy Plan

Planning document only. No work starts until you approve it.

## Goal

Swipe typing in FUTO Keyboard Custom gives wrong words too often, and the official FUTO
app does too, so it isn't a bug in the custom build. The goal is fewer wrong swipes by:

1. making sure the swipe engine runs at full strength on your keyboard (quick check),
2. training a bigger swipe model on your PC (GTX 1660 Super, Linux),
3. adapting that model to *your* swiping style using your own recorded swipes.

A new model replaces FUTO's only if it beats FUTO's complete setup on a fair test (see
"Decision rule").

## What we know

- **Swipe has three parts:** an *encoder* that turns the finger path into letter guesses
  (`honorable_sturgeon`, 0.6M parameters, 2.6 MB), an optional *layout refiner* for
  standard English QWERTY (`magic_macaw`), and an optional small *context model* that
  uses the sentence so far (`chosen_planarian`). A dictionary search on top picks real
  words.
- **FUTO's encoder gets about 84% of test swipes right on its own** (their published
  number). The refiner and context model add to that.
- **There's no better ready-made model.** Every swipe model on Hugging Face was trained
  on the same FUTO data, is about the same size, and scores about the same or worse
  (checked).
- **The training data is free to use:** FUTO's swipe dataset, about 1.17 million real
  swipes, MIT licence.
- **A new encoder can be a drop-in replacement.** FUTO documents its exact inputs and
  outputs (finger path of 64 points; up to 64 key positions; out come letter
  probabilities for 32 time steps). It's stored as an ExecuTorch 1.5 file, the format
  the keyboard's swipe engine loads.
- **FUTO's swipe engine is open source** (swipe-library), so the benchmark can run their
  real engine on a PC, not an imitation.
- **Your setup:** standard QWERTY in portrait, keyboard height set lower than usual.
  Linux PC with a working NVIDIA driver, up to 50 GB of disk, overnight training runs,
  swipes may take up to about 100 ms.

## Decision rule

The custom model is switched on only if it beats **FUTO's full setup** (encoder plus
refiner plus context model) on both:

- FUTO's official test swipes (thousands of words, other people), and
- a held-back part of **your** recorded swipes (never used for training),

and stays under 100 ms per swipe on the phone. Otherwise we keep FUTO's model and say so.

## Stage 0 — Quick check: is swipe running at full strength? (keyboard)

FUTO's layout refiner only switches on if your key positions match a standard QWERTY
keyboard. Your lowered keyboard height may stop it switching on, silently, in both
apps.

- 0.1 Add a **"Swipe engine" status box** to the keyboard's settings: which encoder,
  whether the layout refiner and context model are on, and the time for the last swipe.
  No developer mode needed.
- 0.2 If the refiner is off because of the keyboard height, fix the matching so it
  accepts lowered keyboards (positions are compared after scaling, so height alone
  shouldn't matter; verify against your exact settings).
- 0.3 Unit tests for the layout matching with normal, lowered and raised keyboard
  heights.

**Result:** you'll know whether part of the problem was the refiner being off, and if so
it's fixed before any training.

## Stage 1 — Record your swipes (keyboard, optional, off by default)

- 1.1 Setting **"Record my swipes for training"** (off by default). When on, every swipe
  is saved with the word you *actually kept*:
  - accepted as is → that word,
  - you tapped another suggestion → that word,
  - you deleted it and typed or swiped again → the word you ended up with
    (when this can be worked out reliably; otherwise the swipe is dropped),
  - password fields and incognito fields are never recorded.
- 1.2 Stored **encrypted on the phone** (same approach as the typing history), with a
  count shown in settings, plus **Delete all** and **Export** buttons.
- 1.3 **Export** writes one file to Downloads. You copy it to the PC manually. Nothing is
  ever uploaded.
- 1.4 Also saved per swipe: the key positions of your keyboard at that moment, so
  training sees exactly your keyboard shape.
- 1.5 Unit tests: labelling rules (accept, pick alternative, delete and retype),
  export format, nothing recorded in private fields.

**Result:** after a week or two of normal use you'll have a few thousand real swipes in
your own style, labelled with what you meant.

## Stage 2 — Fair benchmark (built and tested in the cloud session)

- 2.1 Build FUTO's open-source swipe engine (swipe-library) and ExecuTorch 1.5 for
  desktop Linux.
- 2.2 Command-line tool: run any encoder file through the **real** FUTO engine on a set
  of swipes and report the share of words right on the first guess and in the top 3,
  plus time per swipe.
- 2.3 Baseline numbers for FUTO's model: encoder alone, plus refiner, plus refiner and
  context model, on FUTO's test swipes.
- 2.4 The same tool reads your exported swipe file (Stage 1) so the same comparison
  runs on your own swipes.

**Result:** honest target numbers before anything is trained.

## Stage 3 — Training kit (built in the cloud session, smoke-tested on CPU)

A small Python project in a new folder `swipe-training/` in this repo.

- 3.1 `setup.sh`: creates a Python environment with PyTorch (CUDA build for the GTX 1660
  Super) and ExecuTorch 1.5. Checks the GPU is visible.
- 3.2 `prepare_data.py`: downloads FUTO's dataset (about 3–5 GB), converts every swipe
  into the encoder's input format, keeps FUTO's own train/dev/test split.
- 3.3 **Training variations** so the model handles real phones: random scaling
  including **lowered keyboard heights like yours**, small shifts, sloppier and faster
  swipes, different screen sizes.
- 3.4 **Model:** the same kind of network as FUTO's but bigger (several sizes, about
  2M–10M parameters), with the exact same inputs and outputs. Trained to output letter
  probabilities the same way (CTC training).
- 3.5 `train.py`: one command, resumes if interrupted, saves the best checkpoint,
  stays inside 50 GB of disk by deleting old checkpoints.
- 3.6 `evaluate.py`: runs the Stage 2 benchmark against FUTO's model and prints a
  table.
- 3.7 `tune_scoring.py`: tunes the dictionary-search settings for the new model (the
  same settings FUTO tuned for theirs), on the dev split only.
- 3.8 `export.py`: writes the model as an ExecuTorch 1.5 file with the exact inputs and
  outputs FUTO's engine expects, plus its settings file, then **re-loads it and checks**
  it gives the same answers as the training version.
- 3.9 `README.md`: step-by-step instructions for your PC.
- 3.10 Tested here end to end on a tiny slice of the data (CPU), so the overnight run
  works first time.

## Stage 4 — Train on your PC (you)

- 4.1 Run `setup.sh`, then `prepare_data.py`, then `train.py` before bed.
- 4.2 In the morning, `evaluate.py` prints the comparison with FUTO.
- 4.3 You paste me the table. If it doesn't win, I adjust the size or training settings
  and you run again.

## Stage 5 — Adapt to your swiping style (your PC)

- 5.1 `finetune_personal.py`: takes the best model and continues training on your
  exported swipes, mixed with public data so it doesn't forget general English.
- 5.2 A fixed part of your swipes is held back for testing and never trained on.
- 5.3 Can be repeated whenever you've collected more swipes (e.g. monthly).

## Stage 6 — Use the model in the keyboard

- 6.1 **Import swipe model** button (like importing voice models). On import the
  keyboard test-runs the file on a known swipe ("computer") and refuses it if the result
  is wrong.
- 6.2 Switch between **FUTO's model** and **Custom model** at any time.
- 6.3 The custom model uses its own tuned settings from Stage 3.7. FUTO's layout refiner
  only works with FUTO's encoder, so it's switched off for the custom one; the context
  model and dictionary stay.
- 6.4 If the custom model ever fails to load, the keyboard falls back to FUTO's model
  automatically (never crashes; already done for the swipe crash fix).
- 6.5 Status box from Stage 0 shows which model is active and how long swipes take.

## Stage 7 — Check on the phone

- 7.1 Built-in **swipe replay test**: replays your held-back recorded swipes through
  both models on the phone and shows accuracy and time per swipe for each, so the final
  comparison happens on the real device.
- 7.2 Ship the custom model as the default only if it meets the decision rule.

## Optional (only if the main stages leave a gap)

- **Retrain the layout refiner** for the new encoder (it's small, same data). This
  recovers the refiner's extra accuracy for the custom model.
- **Smarter context model** for the final pick between close candidates.
- **int8 version** of the custom model if it's too slow (FUTO's int8 file lost almost
  no accuracy).

## Order and what you do

| Stage | Who | Needs |
|---|---|---|
| 0 Quick check | me | new APK, you look at the status box |
| 1 Recording | me | new APK, you swipe normally for 1–2 weeks |
| 2 Benchmark | me | — |
| 3 Training kit | me | — |
| 4 Train | you | overnight on the PC |
| 5 Personal fine-tune | you | your exported swipes |
| 6 Keyboard import | me | new APK |
| 7 Phone check | you | a few minutes |

Stages 0 and 1 ship first, so recording starts while I build the training kit.

## Risks

- **Gains are not guaranteed.** FUTO's researchers already tuned their model. The bigger
  model may only gain a few points; your-style fine-tuning (Stage 5) is the likelier big
  win.
- **Export compatibility:** the model must use only operations ExecuTorch 1.5 supports
  on phones; Stage 3.8 checks this before you train overnight.
- **GTX 1660 Super:** 6 GB of graphics memory and no fast half-precision mode, which
  limits how big the model can be. Fine for the sizes planned here.
- **Labelling your swipes:** "the word you kept" is sometimes ambiguous (e.g. you
  changed your mind). Unclear cases are dropped rather than guessed.
- **Licences:** FUTO's data is MIT (fine). FUTO's model weights have their own licence,
  but we don't train from them. swipe-library's licence gets checked in Stage 2 before
  it's used.
