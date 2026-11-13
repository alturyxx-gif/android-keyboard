# Tap Accuracy Plan: context-aware key hit areas

Planning document only. No work starts until you approve it.

## Goal

You mostly type by tapping. When a tap lands near the edge between two keys, the keyboard
should pick the letter you most likely meant, using where you tapped, how *you* usually tap,
the word so far, and the sentence so far. This is the idea behind Apple's invisible key
resizing. Fewer wrong letters while typing means less autocorrect, fewer backspaces and fewer
"tge"-style mistakes.

## What the keyboard already does

FUTO has **Smart key-hit detection** (Settings → Text Prediction), on by default:

- After each letter it asks the dictionary which next letters could still make a real word.
- Those keys get a bigger touch area: a tap counts if it's within half a key of their edge,
  instead of going to the nearest key centre.
- It's switched off right after a backspace and on non-letter keys.

Its limits:

| | FUTO now | This plan |
|---|---|---|
| Which keys grow | Any letter that continues *some* word (yes/no) | In proportion to how *likely* each letter is |
| Sentence context | Not used | Typing model (it knows "going to the s…" → *store*) |
| Your vocabulary | Not used | Your typing history and learned words |
| Your touch habits | Not used | Learns where *you* hit each key (e.g. a bit low on the bottom row) |
| Growth | Same for every boosted key | Strong for very likely letters, little for unlikely ones, capped |

After "th", for example, FUTO boosts almost every vowel plus r and w equally, so it can't tell
them apart. Likelihoods can.

## How it decides (the core idea)

For each tap, every nearby key gets a score:

    score(key) = how well the tap fits the key  +  strength × how likely that letter is next

- **How well the tap fits the key:** based on the distance from the key's centre, using *your*
  average offset and spread for that key (Stage 2) instead of an ideal centre.
- **How likely that letter is next:** from the typing model and dictionary (Stage 3).
- The best-scoring key wins, with safety limits:
  - **Centre zone:** a tap in the middle part of a key (inner ~60%) always gives that key.
    Only taps near edges can be changed.
  - **Neighbours only:** a tap can only go to a key that touches the tapped key.
  - **Not after backspace:** like FUTO's, off right after a backspace, and when there's
    nothing to go on (start of a word with no sentence context).
  - **Never in password fields,** on non-letter keys or with accessibility services on.

The letter probabilities are worked out in the background right after each tap, before you
make the next one (taps are usually 150–250 ms apart; the work takes a few milliseconds). If
they aren't ready yet (very fast typing), the keyboard uses FUTO's current behaviour for that
tap. Nothing ever waits.

## Stage 0: Measure first (keyboard)

Without numbers we can't tell whether a change helps.

- 0.1 **"Record my taps for tuning"** (off by default, like swipe recording): for each tapped
  word, saves where each tap landed, which keys were picked, the keyboard shape, the sentence
  before, and the word you ended up with (accepted, corrected, or deleted and retyped).
  Encrypted on the phone, Export / Delete buttons. Never in password or incognito fields.
- 0.2 **Tap statistics** on a new "Tap accuracy" settings page: backspaces per 100 letters,
  share of words autocorrected, share of words you had to fix by hand. Shown for the last
  7 days, so changes are visible over time.
- 0.3 Unit tests for the labelling (same approach as the swipe labeller).

## Stage 1: Replay tester (keyboard + PC kit)

- 1.1 The hit-decision logic as pure, testable Kotlin, separate from Android touch code.
- 1.2 **Replay:** run your recorded taps through any decision rule (FUTO's current one or the
  new one with given settings) and count how many letters come out as the word you meant.
  This gives a "wrong letters per 100" number for each rule on *your* taps.
- 1.3 On the phone: a **Compare** button (like the swipe one). On the PC: a script in a new
  `tap-tuning/` folder that does the same and tries many settings, to pick the best strength
  and safety zone.

## Stage 2: Learn how you tap (keyboard)

- 2.1 From your recorded taps (or, if recording is off, from taps in words you didn't
  correct), keep a running average per key of how far from the centre you hit, and how
  spread out your taps are.
- 2.2 Use that personal centre and spread in "how well the tap fits the key". Starts out as
  the plain key centre and adapts slowly; it needs about 20 taps on a key before it changes
  anything for that key.
- 2.3 Separate profiles for portrait, landscape and one-handed mode (if you use them), since
  you hold the phone differently.
- 2.4 A **Reset** button. Unit tests for the learning (offsets converge, outliers ignored).

This works even without Stage 3, and on its own fixes consistent habits like always hitting
"n" too low.

## Stage 3: Next-letter likelihoods (native + keyboard)

- 3.1 **Check first, on the PC test harness:** feed the typing model sentences with a partly
  typed word and read its prediction for the next letter token (`<CHAR_A>`…`<CHAR_Z>`).
  Confirm it's sensible ("I'm going to the sto" → r). Measure the time per tap on the CPU.
  If the model's letter predictions turn out to be poor, fall back to 3.3 alone.
- 3.2 Native function: given the sentence and the letters typed so far, return 26 letter
  probabilities, reusing what's already been computed (only the new tap is processed).
- 3.3 **Dictionary part:** next-letter likelihoods weighted by word frequency (main dictionary
  plus your typing history and learned words), so common and personal words count more.
- 3.4 Combine model and dictionary, and compute in the background after every tap.
  Roman Urdu sentences: use only the dictionary part plus your Roman Urdu vocabulary (the
  typing model is English).

## Stage 4: Put it together (keyboard)

- 4.1 The new decision rule in key detection, with the safety limits above.
- 4.2 Setting **Smart key-hit detection**: Off / Dictionary (FUTO's) / Context-aware (new),
  plus a **Strength** slider. Default: Context-aware, medium strength (whatever Stage 1
  tuning picks).
- 4.3 **Learns from your corrections:** if you backspace right after the keyboard picked a
  different key than the one under your finger, that counts against it, and the strength is
  lowered a little. If that happens a lot, it backs off automatically.
- 4.4 Optional **debug view**: draws the current invisible key areas on the keyboard, so you
  can see it working while testing.
- 4.5 Tests: decision rule (centre zone respected, neighbours only, off after backspace, off
  in password fields), and that it never slows typing (background work is skipped if late).

## Stage 5: Check it on your phone

- 5.1 Compare on your recorded taps (Stage 1) between Off, FUTO's dictionary boost, and
  Context-aware.
- 5.2 A week of normal use, then compare the tap statistics (Stage 0.2) with the week before.
- 5.3 Keep it on only if wrong letters and backspaces go down. Otherwise switch back to FUTO's
  setting and say so.

## Order and what you do

| Stage | Who | Needs from you |
|---|---|---|
| 0 Measure | me | new APK, turn on tap recording, type normally |
| 1 Replay tester | me | — |
| 2 Your touch habits | me | keeps learning as you type |
| 3 Letter likelihoods | me | — |
| 4 Put together | me | new APK, try it, say how it feels |
| 5 Check | you + me | Compare button + a week of use |

Stages 0–1 ship first so recording starts while the rest is built. No PC training is needed;
the PC script in 1.3 only tunes settings (minutes).

## Risks

- **"It typed a letter I didn't press."** The biggest risk with any key resizing. Mitigated
  by the centre zone, neighbours only, the backspace feedback in 4.3 and the strength slider.
  Taps in the middle of a key are never changed.
- **Speed:** the next-letter work must finish between taps. It runs in the background and
  is skipped when late, so typing never waits; at worst you get FUTO's current behaviour.
- **Typing model's letter predictions** may be less useful than hoped; Stage 3.1 checks
  this before anything depends on it.
- **Measuring "what you meant"** is approximate (same rules as swipe recording: unclear
  cases are dropped).
- **Battery:** a few milliseconds of typing-model work per tap, on a model that already runs
  for suggestions. Expected to be negligible; Stage 0.2's stats page will show the time per
  tap so we can check.

## Questions for you before starting

1. **Debug view (4.4):** do you want to see the invisible key areas while testing, or keep the
   keyboard looking exactly as it does now?
2. **Landscape / one-handed:** do you type in those modes often enough to need separate
   touch profiles, or is portrait enough?
3. **Roman Urdu:** you said English words appear in Roman Urdu sentences. Is it enough for
   Roman Urdu sentences to use your Roman Urdu vocabulary plus the dictionary, without the
   English typing model?
