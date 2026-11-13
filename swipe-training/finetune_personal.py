#!/usr/bin/env python3
"""
Adapt a trained model to your own swiping style, using swipes exported from the keyboard
(Settings > Swipe > Swipe model & training > Export).

    python finetune_personal.py runs/medium/best.pt ~/futo-swipes-20261201-2130.jsonl.gz
    python finetune_personal.py runs/medium/best.pt export1.jsonl.gz export2.jsonl.gz  # several

Your swipes are split by date: the oldest 75% are trained on, the next 10% pick the best epoch,
and the newest 15% are never trained on (evaluate.py --personal uses them to compare models).
Public data is mixed in so the model doesn't forget how other words look.
Result: runs/personal/best.pt -> python export.py runs/personal/best.pt --name mine
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset

sys.path.insert(0, str(Path(__file__).parent))
from swipekit.data import AugmentConfig, parse_personal_line  # noqa: E402
from swipekit.personal import load_lines, split  # noqa: E402
from swipekit.training import SwipeSet, load_model, train  # noqa: E402

# Your swipes are already in the phone's exact keyboard frame: vary them less than public data
PERSONAL_AUG = AugmentConfig(p=0.6, scale_x=(0.95, 1.05), scale_y=(0.9, 1.1), shift=0.015,
                             rotate_deg=1.5, key_jitter=0.004, point_noise=0.004, wobble=0.012)


def personal_set(lines, train_mode):
    samples = [s for s in (parse_personal_line(l) for l in lines) if s is not None]
    return SwipeSet([s.feats for s in samples], [s.word for s in samples],
                    keys=[s.keys for s in samples], letters=[s.letters for s in samples],
                    train=train_mode, aug=PERSONAL_AUG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="model to start from (e.g. runs/medium/best.pt)")
    ap.add_argument("swipes", nargs="+", help="exported swipe files (.jsonl.gz)")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/personal")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=4, help="copies of your swipes per epoch")
    ap.add_argument("--public-ratio", type=float, default=2.0, help="public swipes per personal swipe")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    lines = load_lines(args.swipes)
    tr, va, te = split(lines)
    print(f"Your swipes: {len(lines)} (train {len(tr)}, validation {len(va)}, held-out test {len(te)})")
    if len(tr) < 200:
        print("Fewer than 200 swipes to train on - record for longer first (a few thousand is ideal).")
        return

    mine_train = personal_set(tr, True)
    mine_val = personal_set(va, False)
    pub = np.load(Path(args.data) / "train.npz", mmap_mode="r")
    n_pub = int(len(mine_train) * args.repeat * args.public_ratio)
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(len(pub["words"]), size=min(n_pub, len(pub["words"])), replace=False))
    public = SwipeSet(pub["feats"][pick], [str(w) for w in pub["words"][pick]], train=True)
    train_set = ConcatDataset([mine_train] * args.repeat + [public])

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_model(args.checkpoint, device).train()
    best = train(model, train_set, mine_val, Path(args.out), epochs=args.epochs, batch=args.batch,
                 lr=args.lr, device=device, workers=args.workers, warmup_frac=0.05, ema_decay=0.995,
                 resume=False, note=f"personal fine-tune of {args.checkpoint} on {len(tr)} swipes")
    print(f"\nFinished. Next:\n  python export.py {best} --name mine\n"
          f"  python evaluate.py exports/mine --personal {' '.join(args.swipes)}")


if __name__ == "__main__":
    main()
