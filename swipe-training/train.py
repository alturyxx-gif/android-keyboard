#!/usr/bin/env python3
"""
Train a swipe encoder on FUTO's public data (after prepare_data.py).

    python train.py --size medium            # the recommended overnight run
    python train.py --size small --epochs 20 # quicker
    python train.py --size tiny --epochs 1 --limit 2000   # smoke test

Interrupted? Run the same command again: it continues from the last finished epoch.
Result: runs/<size>/best.pt  (then run export.py and evaluate.py)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from swipekit.model import SwipeEncoder, ModelConfig  # noqa: E402
from swipekit.training import SwipeSet, train  # noqa: E402

EPOCHS = {"tiny": 1, "small": 25, "medium": 30, "large": 30}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="medium", choices=["tiny", "small", "medium", "large"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default=None, help="default runs/<size>")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="use only this many training swipes (testing)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cpu" and args.size != "tiny":
        print("WARNING: no NVIDIA GPU found by PyTorch - training will be very slow. Check setup.sh output.")
    data = Path(args.data)
    tr = np.load(data / "train.npz", mmap_mode="r")
    dv = np.load(data / "dev.npz", mmap_mode="r")
    feats, words = tr["feats"], [str(w) for w in tr["words"]]
    if args.limit:
        feats, words = feats[: args.limit], words[: args.limit]
    dev_n = 20000 if not args.limit else min(2000, len(dv["words"]))
    train_set = SwipeSet(feats, words, train=True)
    dev_set = SwipeSet(dv["feats"][:dev_n], [str(w) for w in dv["words"][:dev_n]])

    torch.manual_seed(0)
    model = SwipeEncoder(ModelConfig.preset(args.size))
    out = Path(args.out or f"runs/{args.size}")
    best = train(model, train_set, dev_set, out,
                 epochs=args.epochs or EPOCHS[args.size], batch=args.batch, lr=args.lr,
                 device=device, workers=args.workers, note=f"size={args.size}")
    print(f"\nFinished. Best model: {best}\nNext: python export.py {best}")


if __name__ == "__main__":
    main()
