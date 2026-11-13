#!/usr/bin/env python3
"""
Downloads FUTO's public swipe dataset (MIT licence) and converts it into compact training files.

    python prepare_data.py                     # everything (about 6 GB download)
    python prepare_data.py --limit 20000       # small slice, for a quick test

Output (in --out, default data/):
    train.npz  dev.npz  test.npz   paths prepared exactly like the phone's swipe engine
    dev.jsonl  test.jsonl          raw lines for the engine benchmark (evaluate.py)

The train split is FUTO's train.jsonl plus the English QWERTY single-finger swipes from
swipe-2..5. dev and test are FUTO's own dev/test splits and are never trained on.
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from swipekit.data import parse_dataset_line, QWERTY  # noqa: E402

BASE = "https://huggingface.co/datasets/futo-org/swipe.futo.org/resolve/main/"
TRAIN_FILES = ["train.jsonl", "swipe-2/swipe2.jsonl", "swipe-3/swipe3.jsonl",
               "swipe-4/swipe4.jsonl", "swipe-5/swipe5.jsonl"]


def download(name: str, dest: Path):
    """Download with resume (re-running continues an interrupted download)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    headers = {"User-Agent": "swipe-training-kit/1.0"}
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(BASE + name, headers=headers)
    try:
        resp = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # already complete
            return
        raise
    total = have + int(resp.headers.get("Content-Length", 0))
    mode = "ab" if have and resp.status == 206 else "wb"
    if mode == "wb":
        have = 0
    with open(dest, mode) as f:
        last = time.time()
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            have += len(chunk)
            if time.time() - last > 5:
                print(f"  {name}: {have / 1e6:.0f} / {total / 1e6:.0f} MB", flush=True)
                last = time.time()
    print(f"  {name}: done ({have / 1e6:.0f} MB)")


def _parse(line):
    s = parse_dataset_line(line)
    if s is None:
        return None
    return s.feats.astype(np.float16), s.word, " ".join(s.context)


def convert(paths, out: Path, limit: int, workers: int, check_alignment=True):
    feats, words, ctxs = [], [], []
    for p in paths:
        if not p.exists():
            print(f"  {p}: not found, skipped")
            continue
        before = len(words)
        with open(p) as f:
            lines = f if limit <= 0 else (l for _, l in zip(range(limit), f))
            with Pool(workers) as pool:
                for r in pool.imap(_parse, lines, chunksize=256):
                    if r is not None:
                        feats.append(r[0]); words.append(r[1]); ctxs.append(r[2])
        n = len(words) - before
        if check_alignment and n:
            # Sanity check: paths should start near the word's first letter
            f = np.stack(feats[before:before + min(n, 2000)]).astype(np.float32)
            first = np.array([QWERTY[w[0]] for w in words[before:before + min(n, 2000)]])
            d = float(np.linalg.norm(f[:, :, 0] - first, axis=1).mean())
            status = "ok" if d < 0.12 else "LOOKS MISALIGNED - excluded"
            print(f"  {p.name}: {n} swipes, start-to-first-letter distance {d:.3f} ({status})")
            if d >= 0.12:
                del feats[before:], words[before:], ctxs[before:]
    np.savez(out, feats=np.stack(feats), words=np.array(words), context=np.array(ctxs))
    print(f"-> {out}: {len(words)} swipes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--raw", default="data/raw", help="where downloaded files are kept")
    ap.add_argument("--limit", type=int, default=0, help="max lines per file (0 = all)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.skip_download:
        print("Downloading FUTO's swipe dataset (resumes if interrupted)...")
        for name in ["dev.jsonl", "test.jsonl"] + TRAIN_FILES:
            download(name, raw / name)

    print("Preparing dev and test splits...")
    convert([raw / "dev.jsonl"], out / "dev.npz", args.limit, args.workers)
    convert([raw / "test.jsonl"], out / "test.npz", args.limit, args.workers)
    for split in ["dev", "test"]:
        # Raw lines for the engine benchmark (first 10k of each)
        with open(raw / f"{split}.jsonl") as f, open(out / f"{split}.jsonl", "w") as g:
            for _, l in zip(range(10000), f):
                g.write(l)
    print("Preparing train split...")
    convert([raw / n for n in TRAIN_FILES], out / "train.npz", args.limit, args.workers)
    print("Done.")


if __name__ == "__main__":
    main()
