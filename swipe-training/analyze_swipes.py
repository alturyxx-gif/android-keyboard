#!/usr/bin/env python3
"""
What kind of swipe mistakes are you getting? Reads the swipes you exported from the keyboard
(Swipe model & training > Export) and sorts them by what the keyboard did at the time.
Runs only on your PC; nothing is sent anywhere.

    python analyze_swipes.py ~/futo-swipes-*.jsonl.gz
    python analyze_swipes.py ~/futo-swipes-*.jsonl.gz --top 40     # longer lists
    python analyze_swipes.py ~/futo-swipes-*.jsonl.gz --no-words   # numbers only

Kinds of mistakes, and what fixes them:
  ranking     the right word was offered, but not first   -> context / word frequency
  misread     the right word wasn't offered at all, but it's in the dictionary
                                                           -> personal fine-tune of the swipe model
  not a word  the right word isn't in the dictionary       -> add it to your dictionary
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from swipekit import engine as E  # noqa: E402
from swipekit.personal import load_lines  # noqa: E402


def norm(w: str) -> str:
    return "".join(c for c in w.lower() if c.isalpha())


def load_dictionary(path: Path) -> set:
    words = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("word="):
                words.add(norm(line.split(",", 1)[0][5:]))
    return words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("swipes", nargs="+")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--no-words", action="store_true", help="don't print any words")
    ap.add_argument("--cache", default="cache")
    args = ap.parse_args()

    lines = load_lines(args.swipes)
    if not lines:
        print("No swipes found in those files.")
        return
    dictionary = load_dictionary(E.vocab(Path(args.cache)))

    by_model = Counter()
    kinds = Counter()
    ranking, misread, oov = Counter(), Counter(), Counter()
    for line in lines:
        d = json.loads(line)
        word = norm(d.get("word", ""))
        cands = [norm(c) for c in d.get("candidates", [])]
        if not word or not cands:
            continue
        frame = d.get("frame") or {}
        by_model["with refiner (FUTO's model)" if frame.get("refiner") else "without refiner (custom model, or FUTO's without refiner)"] += 1
        if cands[0] == word:
            kinds["right first time"] += 1
        elif word in cands:
            kinds["ranking"] += 1
            ranking[(cands[0], word)] += 1
        elif word not in dictionary:
            kinds["not a word"] += 1
            oov[word] += 1
        else:
            kinds["misread"] += 1
            misread[(cands[0], word)] += 1

    n = sum(kinds.values())
    print(f"{n} labelled swipes")
    for k, v in by_model.items():
        print(f"  {v} {k}")
    print()
    rows = [
        ("right first time", "-"),
        ("ranking", "right word offered but not first -> context / word frequency"),
        ("misread", "right word not offered, but in dictionary -> personal fine-tune"),
        ("not a word", "not in the dictionary -> add to your dictionary"),
    ]
    for k, fix in rows:
        print(f"  {k:18s} {kinds[k]:6d}  {kinds[k] * 100 / n:5.1f}%   {fix}")
    errors = n - kinds["right first time"]
    if errors:
        print(f"\nOf your {errors} mistakes: "
              + ", ".join(f"{kinds[k] * 100 / errors:.0f}% {k}" for k in ("ranking", "misread", "not a word")))

    if args.no_words:
        return
    def show(title, counter, fmt):
        if counter:
            print(f"\n{title}")
            for item, c in counter.most_common(args.top):
                print(f"  {c:4d}x  {fmt(item)}")
    show("Most common ranking mistakes (keyboard put first -> what you meant):", ranking, lambda p: f"{p[0]} -> {p[1]}")
    show("Most common misreads (keyboard put first -> what you meant):", misread, lambda p: f"{p[0]} -> {p[1]}")
    show("Words not in the dictionary (add these to your personal dictionary):", oov, lambda w: w)
    print("\nNote: swipes are only recorded when the keyboard could tell which word you meant "
          "(accepted, picked from the strip, or deleted and retyped right away).")


if __name__ == "__main__":
    main()
