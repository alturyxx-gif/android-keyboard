# Swipe model training kit

Trains a swipe model for FUTO Keyboard Custom on your own PC (Linux, NVIDIA GPU), compares it
with FUTO's model using FUTO's real swipe engine, and makes a file you import on the phone.
See `../SWIPE_PLAN.md` for the why.

Everything stays local: the only downloads are FUTO's public swipe dataset (MIT licence),
FUTO's swipe models (for the comparison) and the English dictionary the keyboard already uses.

## What you need

- Linux with a working NVIDIA driver (`nvidia-smi` shows your GTX 1660 Super)
- Python 3.10–3.13, git, cmake, a C++ compiler:
  - Ubuntu/Debian: `sudo apt install python3-venv python3-dev git cmake build-essential`
  - Arch: `sudo pacman -S --needed base-devel git cmake uv`. Arch's own Python is usually too
    new for ExecuTorch 1.2, so get 3.13 next to it with `uv python install 3.13`, then run setup as
    `PYTHON=$(uv python find 3.13) ./setup.sh`
- About 20 GB free disk space

## 1. Setup (once, 10–40 minutes)

```bash
cd swipe-training
./setup.sh
source .venv/bin/activate
```

`setup.sh` installs PyTorch with CUDA and ExecuTorch 1.2 (the version inside the keyboard's swipe
engine), and builds FUTO's swipe engine for the comparison. It prints whether PyTorch can see
the GPU. If it says `CUDA available: False`, training won't use the GPU, so stop and check the driver.

## 2. Get the data (once, about 6 GB download + a few minutes)

```bash
python prepare_data.py
```

Downloads resume if interrupted (just run it again). It checks that every part of the dataset
lines up with the keyboard layout and skips any part that doesn't.

## 3. Train (overnight)

```bash
python train.py --size medium
```

| Size | Parameters | vs FUTO's model | Rough time on a GTX 1660 Super |
|---|---|---|---|
| `small` | 1.4M | 2× | 2–3 hours |
| `medium` (recommended) | 3.7M | 6× | 4–6 hours |
| `large` | 9.7M | 15× | 8–12 hours |

The times are estimates. The first epoch prints a real "epoch ends in" time, so check it after
15 minutes. Interrupted? Run the same command again and it continues from the last finished epoch.
Progress is printed every 200 steps. After each epoch it prints the "dev loss" (lower is better).

## 4. Export and compare

```bash
python export.py runs/medium/best.pt
python evaluate.py exports/medium --tune
```

`export.py` writes the phone file and checks that it only uses operations the phone's swipe engine
supports, and that it gives the same answers as in training.
`evaluate.py --tune` picks the best search settings, then prints a table like:

```
                                          top-1    top-4    time/swipe (PC)
FUTO test swipes:
  FUTO: model + refiner + context         91.8%    95.0%      3.8 ms
  FUTO: model + context (no refiner)      90.6%    94.6%      3.0 ms
  Yours (medium) + context                 ...
```

**top-1** = the right word was the first guess. Paste the table to Claude.
Your model is only worth using if it beats the first row.

## 5. Put it on the phone

Copy `exports/medium/swipe-model-medium.zip` to the phone (USB, KDE Connect, anything), then:
**Settings → Swipe → Swipe model & training → Import model**, and turn on
**Use my custom swipe model**. The keyboard test-runs the model before accepting it, and goes back
to FUTO's model automatically if yours ever fails to load.

## 6. Adapt it to your swiping (after 1–2 weeks of recording)

On the phone: **Swipe model & training → Record my swipes for training** (on), swipe normally,
then **Export**. Copy the `futo-swipes-….jsonl.gz` file from Downloads to the PC, then delete
it from the phone (it isn't encrypted). Then:

```bash
python finetune_personal.py runs/medium/best.pt ~/futo-swipes-*.jsonl.gz
python export.py runs/personal/best.pt --name mine
python evaluate.py exports/mine --tune --personal ~/futo-swipes-*.jsonl.gz
```

This takes minutes, not hours. The newest 15% of your swipes are never trained on, and
`evaluate.py --personal` compares FUTO's model and yours on them. On the phone,
**Compare models** does the same comparison using the keyboard's own dictionary.
Repeat whenever you've collected more swipes. Export files can be combined (pass several).

## What kind of mistakes are you getting?

```bash
python analyze_swipes.py ~/futo-swipes-*.jsonl.gz
```

Sorts your recorded swipes into: right first time; **ranking** (the right word was offered but
not first, so better context or word frequency would fix it); **misread** (the right word wasn't
offered but is in the dictionary, so the personal fine-tune fixes it); **not a word** (not in the
dictionary, so add it to your personal dictionary). It also lists your most common mistakes.
It runs locally; add `--no-words` to print numbers only.

## Files

| | |
|---|---|
| `setup.sh` | one-time setup |
| `prepare_data.py` | download and convert FUTO's dataset |
| `train.py` | train on public data |
| `finetune_personal.py` | adapt to your exported swipes |
| `export.py` | make the phone file, with compatibility checks |
| `evaluate.py` | compare with FUTO through FUTO's real engine |
| `analyze_swipes.py` | what kind of mistakes your recorded swipes show |
| `swipekit/` | model, data handling, training loop, engine wrapper |

## Notes

- The model has the same inputs and outputs as FUTO's encoder (`honorable_sturgeon`), so the
  keyboard's swipe engine uses it unchanged. It works with any key positions: keys are scored
  by their coordinates, not as fixed outputs.
- Paths are prepared exactly like the swipe engine does (60 Hz, then 64 points). The training
  variations include shorter and taller keyboards, like your lowered keyboard height.
- FUTO's layout refiner only works with FUTO's encoder, so the keyboard switches it off for a
  custom model. The context model and dictionary still apply.
