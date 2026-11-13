"""Training loop shared by train.py and finetune_personal.py."""
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data import AugmentConfig, QWERTY_KEYS, LETTERS, augment, pad_keys
from .model import SwipeEncoder, ModelConfig, T_OUT, MAX_KEYS, count_params

BLANK = MAX_KEYS


def fits_ctc(word: str) -> bool:
    """CTC needs one output step per letter, plus one between repeated letters."""
    repeats = sum(1 for a, b in zip(word, word[1:]) if a == b)
    return 1 <= len(word) and len(word) + repeats <= T_OUT


class SwipeSet(Dataset):
    """Public dataset (all QWERTY) and/or personal samples with their own keys."""

    def __init__(self, feats, words, keys=None, letters=None, train=False, aug=None, seed=0):
        idx = [i for i, w in enumerate(words) if fits_ctc(w)]
        self.feats = feats
        self.words = words
        self.idx = idx
        self.keys = keys          # None = QWERTY for all, else list of [K,2]
        self.letters = letters    # None = a..z for all, else list of strings
        self.train = train
        self.aug = aug or AugmentConfig()
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        f = np.asarray(self.feats[j], dtype=np.float32)
        keys = QWERTY_KEYS if self.keys is None else self.keys[j]
        letters = LETTERS if self.letters is None else self.letters[j]
        if self.train:
            f, keys = augment(f, keys, self.aug, self.rng)
        k, m = pad_keys(keys)
        tgt = [letters.index(c) for c in self.words[j]]
        return f, k, m, tgt


def collate(batch):
    f = torch.from_numpy(np.stack([b[0] for b in batch]))
    k = torch.from_numpy(np.stack([b[1] for b in batch]))
    m = torch.from_numpy(np.stack([b[2] for b in batch]))
    tl = torch.tensor([len(b[3]) for b in batch], dtype=torch.long)
    tg = torch.tensor([t for b in batch for t in b[3]], dtype=torch.long)
    return f, k, m, tg, tl


def ctc_loss(log_em, targets, target_lengths):
    T = log_em.shape[1]
    il = torch.full((log_em.shape[0],), T, dtype=torch.long)
    return F.ctc_loss(log_em.transpose(0, 1).float(), targets, il, target_lengths,
                      blank=BLANK, zero_infinity=True, reduction="mean")


def greedy(log_em, mask_letters):
    """Greedy CTC decode to slot indices (a rough accuracy measure; the engine searches
    the dictionary, which is much better)."""
    best = log_em.argmax(-1).cpu().numpy()
    out = []
    for row in best:
        prev, seq = -1, []
        for c in row:
            if c != prev and c != BLANK:
                seq.append(int(c))
            prev = c
        out.append(seq)
    return out


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
            else:
                s.copy_(p)


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    model.eval()
    total, n, exact = 0.0, 0, 0
    for bi, (f, k, m, tg, tl) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        f, k, m = f.to(device), k.to(device), m.to(device)
        le = model(f, k, m)[0]
        total += float(ctc_loss(le, tg, tl)) * f.shape[0]
        n += f.shape[0]
        pos = 0
        for seq, L in zip(greedy(le, None), tl.tolist()):
            exact += int(seq == tg[pos:pos + L].tolist())
            pos += L
    model.train()
    return total / max(1, n), exact / max(1, n)


def train(model, train_set, dev_set, out_dir: Path, *, epochs, batch, lr, device,
          workers=4, amp=True, log_every=200, resume=True, weight_decay=0.05,
          warmup_frac=0.03, ema_decay=0.999, keep=2, note=""):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    dl = DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=workers,
                    collate_fn=collate, drop_last=True, persistent_workers=workers > 0,
                    pin_memory=device.type == "cuda")
    dev = DataLoader(dev_set, batch_size=batch, shuffle=False, num_workers=max(1, workers // 2),
                     collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))
    total_steps = epochs * len(dl)
    warm = max(1, int(total_steps * warmup_frac))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = EMA(model, ema_decay)
    start_epoch, best = 0, float("inf")
    history = []

    last = out_dir / "last.pt"
    if resume and last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); ema.shadow.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_epoch, best, history = ck["epoch"] + 1, ck["best"], ck.get("history", [])
        print(f"Resumed from epoch {start_epoch} (best dev loss {best:.4f})")

    print(f"Model: {count_params(model) / 1e6:.2f}M parameters; {len(train_set)} training swipes; "
          f"{len(dl)} steps per epoch; device {device}{' (mixed precision)' if use_amp else ''}")
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        model.train()
        run = 0.0
        for step, (f, k, m, tg, tl) in enumerate(dl):
            f, k, m = f.to(device, non_blocking=True), k.to(device, non_blocking=True), m.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                le = model(f, k, m)[0]
            loss = ctc_loss(le, tg, tl)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)
            run += float(loss)
            if (step + 1) % log_every == 0:
                el = time.time() - t0
                eta = el / (step + 1) * (len(dl) - step - 1)
                print(f"  epoch {epoch + 1}/{epochs} step {step + 1}/{len(dl)} loss {run / log_every:.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} ({el / 60:.1f} min, epoch ends in {eta / 60:.1f} min)",
                      flush=True)
                run = 0.0
        dl_loss, dl_acc = evaluate(ema.shadow, dev, device)
        mins = (time.time() - t0) / 60
        history.append({"epoch": epoch + 1, "dev_loss": dl_loss, "dev_greedy_acc": dl_acc, "minutes": mins})
        print(f"Epoch {epoch + 1}: dev loss {dl_loss:.4f}, greedy word accuracy {dl_acc * 100:.1f}% "
              f"(the engine's dictionary search does much better), {mins:.1f} min", flush=True)
        ck = {"model": model.state_dict(), "ema": ema.shadow.state_dict(), "opt": opt.state_dict(),
              "sched": sched.state_dict(), "epoch": epoch, "best": min(best, dl_loss),
              "config": model.cfg.to_dict(), "history": history, "note": note}
        torch.save(ck, out_dir / "last.pt.tmp"); (out_dir / "last.pt.tmp").replace(last)
        if dl_loss < best:
            best = dl_loss
            torch.save({"ema": ema.shadow.state_dict(), "config": model.cfg.to_dict(),
                        "epoch": epoch + 1, "dev_loss": dl_loss, "history": history, "note": note},
                       out_dir / "best.pt")
            print(f"  new best -> {out_dir / 'best.pt'}")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    return out_dir / "best.pt"


def load_model(path, device="cpu") -> SwipeEncoder:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = SwipeEncoder(ModelConfig(**ck["config"]))
    m.load_state_dict(ck["ema"] if "ema" in ck else ck["model"])
    return m.to(device).eval()
