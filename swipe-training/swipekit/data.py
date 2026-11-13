"""
Data handling: reading FUTO's swipe dataset and your exported swipes, preparing paths exactly
like the phone's swipe engine does, and the training-time variations.
"""
import json
import math
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .model import T_IN, MAX_KEYS

LETTERS = "abcdefghijklmnopqrstuvwxyz"

# FUTO's en_qwerty layout (swipe-library models/layouts/en_qwerty.json): the frame the public
# dataset is recorded in. Rows at 1/6, 1/2, 5/6; q at x=0.05, p at 0.95.
_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
_OFFS = [0.0, 0.5, 1.5]
QWERTY = {}
for _r, (_row, _off) in enumerate(zip(_ROWS, _OFFS)):
    for _i, _c in enumerate(_row):
        QWERTY[_c] = ((_off + _i + 0.5) / 10.0, (_r + 0.5) / 3.0)
QWERTY_KEYS = np.array([QWERTY[c] for c in LETTERS], dtype=np.float32)  # [26, 2]

# Newer dataset files with {"L": ..., "R": ...} data have y scaled differently; the engine's
# own evaluation tool multiplies y by this (swipe-library targets/infer_jsonl.cpp)
LR_Y_SCALE = 1.401709402


# ----------------------------------------------------------------------------------------
# Preprocessing identical to swipe-library (src/resampler.cpp + SwipeEngine::predict_segment)
# ----------------------------------------------------------------------------------------

def _interp_lower_bound(t, x, y, target):
    """Linear interpolation using std::lower_bound semantics, as in resampler.cpp."""
    n = len(t)
    idx1 = np.searchsorted(t, target, side="left")
    out_x = np.empty_like(target)
    out_y = np.empty_like(target)
    first = idx1 == 0
    last = idx1 >= n
    mid = ~(first | last)
    out_x[first], out_y[first] = x[0], y[0]
    out_x[last], out_y[last] = x[n - 1], y[n - 1]
    i1 = idx1[mid]
    i0 = i1 - 1
    seg = t[i1] - t[i0]
    frac = np.where(seg > 1e-12, (target[mid] - t[i0]) / np.where(seg > 1e-12, seg, 1.0), 0.0)
    out_x[mid] = x[i0] + frac * (x[i1] - x[i0])
    out_y[mid] = y[i0] + frac * (y[i1] - y[i0])
    return out_x, out_y


def resample_60hz(x, y, t):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    t = np.asarray(t, dtype=np.float32)
    n = len(x)
    if n == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    if n == 1:
        return np.array([x[0], x[0]], np.float32), np.array([y[0], y[0]], np.float32)
    duration = float(t[-1] - t[0])
    if duration <= 1e-12:
        return np.array([x[0], x[-1]], np.float32), np.array([y[0], y[-1]], np.float32)
    num = max(2, int(round(duration / (1000.0 / 60.0))) + 1)
    step = np.float32(duration / (num - 1))
    target = (t[0] + np.arange(num, dtype=np.float32) * step).astype(np.float32)
    target = np.minimum(target, t[-1])
    return _interp_lower_bound(t, x, y, target)


def resample_linear(x, y, length=T_IN):
    n = len(x)
    if n == 0:
        return np.zeros(length, np.float32), np.zeros(length, np.float32)
    if n == 1:
        return np.full(length, x[0], np.float32), np.full(length, y[0], np.float32)
    idx = np.arange(length, dtype=np.float32) / np.float32(length - 1) * np.float32(n - 1)
    i0 = idx.astype(np.int64)
    i1 = np.minimum(i0 + 1, n - 1)
    frac = idx - i0
    return (x[i0] + frac * (x[i1] - x[i0])).astype(np.float32), (y[i0] + frac * (y[i1] - y[i0])).astype(np.float32)


def engine_features(x, y, t) -> np.ndarray:
    """[2, 64] features exactly as the swipe engine computes them (60 Hz, 64 points, clamp)."""
    hx, hy = resample_60hz(x, y, t)
    rx, ry = resample_linear(hx, hy)
    return np.stack([np.clip(rx, 0, 1), np.clip(ry, 0, 1)]).astype(np.float32)


# ----------------------------------------------------------------------------------------
# Records
# ----------------------------------------------------------------------------------------

@dataclass
class Sample:
    feats: np.ndarray          # [2, 64] float32, model frame
    keys: np.ndarray           # [K, 2] float32, letter key centres in the same frame
    letters: str               # letters of `keys`, in slot order
    word: str                  # target word (letters only, lower case)
    context: list              # previous words in the sentence (for evaluation)


def normalize_word(word: str) -> str:
    """Lower case, letters only (like the engine's evaluation); '' if it has other letters."""
    w = "".join(c for c in word.lower() if c.isalpha())
    return w if w and all(c in LETTERS for c in w) else ""


def parse_dataset_line(line: str, layout_keys: Optional[dict] = None) -> Optional[Sample]:
    """One line of FUTO's swipe.futo.org dataset -> Sample (single-finger English QWERTY only)."""
    try:
        d = json.loads(line)
    except Exception:
        return None
    if d.get("language", "en") != "en":
        return None
    lay = d.get("layout", "qwerty")
    if lay not in ("qwerty", "en_qwerty"):
        return None
    word = normalize_word(d.get("word", ""))
    if len(word) < 2:
        return None
    data = d.get("data")
    if isinstance(data, list):
        pts = data
        yscale = 1.0
    elif isinstance(data, dict):
        segs = (data.get("L") or []) + (data.get("R") or [])
        if len(segs) != 1:
            return None  # two-finger swipes: not used
        pts = segs[0]
        yscale = LR_Y_SCALE
    else:
        return None
    if len(pts) < 3:
        return None
    try:
        t0 = float(pts[0]["t"])
        x = [float(p["x"]) for p in pts]
        y = [float(p["y"]) * yscale for p in pts]
        t = [float(p["t"]) - t0 for p in pts]
    except Exception:
        return None
    keys = QWERTY_KEYS if layout_keys is None else np.array([layout_keys[c] for c in LETTERS], np.float32)
    ctx = []
    if "sentence" in d and "word_idx" in d:
        ctx = str(d["sentence"]).split()[: int(d["word_idx"])]
    return Sample(engine_features(x, y, t), keys, LETTERS, word, ctx)


def parse_personal_line(line: str) -> Optional[Sample]:
    """One line of the keyboard's exported swipes (futo-swipes-*.jsonl.gz) -> Sample.

    Uses the exact frame the keyboard gave the swipe engine (`frame`), so the model is trained
    on what it will see on the phone.
    """
    try:
        d = json.loads(line)
    except Exception:
        return None
    word = normalize_word(d.get("word", ""))
    if len(word) < 2:
        return None
    path, frame = d.get("path") or {}, d.get("frame")
    x, y, t = path.get("x") or [], path.get("y") or [], path.get("t") or []
    if len(x) < 3 or not frame or not frame.get("letters"):
        return None
    sx, sy, ox, oy = frame["sx"], frame["sy"], frame["ox"], frame["oy"]
    # Same transform as SwipeDecoderDictionary.transformSegment (y uses the 4/3 factor)
    mx = [xi * sx + ox for xi in x]
    my = [min(1.0, yi * (4.0 / 3.0) * sy + oy) for yi in y]
    letters = frame["letters"]
    if not all(c in letters for c in word):
        return None
    keys = np.stack([np.array(frame["xs"], np.float32), np.array(frame["ys"], np.float32)], axis=1)
    return Sample(engine_features(mx, my, t), keys, letters, word, list(d.get("context") or []))


def to_engine_jsonl(sample_line: str) -> Optional[str]:
    """Personal export line -> the JSONL format swipe-library's infer_jsonl tool reads."""
    s = parse_personal_line(sample_line)
    if s is None:
        return None
    d = json.loads(sample_line)
    frame = d["frame"]
    p = d["path"]
    pts = [{"x": xi * frame["sx"] + frame["ox"],
            "y": min(1.0, yi * (4.0 / 3.0) * frame["sy"] + frame["oy"]),
            "t": ti} for xi, yi, ti in zip(p["x"], p["y"], p["t"])]
    ctx = list(d.get("context") or [])
    return json.dumps({"word": d["word"], "data": pts,
                       "sentence": " ".join(ctx + [d["word"]]), "word_idx": len(ctx)})


# ----------------------------------------------------------------------------------------
# Training-time variations
# ----------------------------------------------------------------------------------------

@dataclass
class AugmentConfig:
    p: float = 0.8              # chance a sample is varied at all
    scale_x: tuple = (0.85, 1.12)
    scale_y: tuple = (0.70, 1.30)   # keyboards shorter or taller than the dataset's (like yours)
    shift: float = 0.04
    rotate_deg: float = 3.0
    key_jitter: float = 0.01    # keys not exactly where the dataset had them
    point_noise: float = 0.006  # finger jitter
    wobble: float = 0.02        # smooth sloppy deviation along the path


def augment(feats: np.ndarray, keys: np.ndarray, cfg: AugmentConfig, rng: random.Random):
    """Applies the same random keyboard transform to the path and the keys, plus noise."""
    if rng.random() > cfg.p:
        return feats, keys
    sx = rng.uniform(*cfg.scale_x)
    sy = rng.uniform(*cfg.scale_y)
    a = math.radians(rng.uniform(-cfg.rotate_deg, cfg.rotate_deg))
    ca, sa = math.cos(a), math.sin(a)
    m = np.array([[sx * ca, -sy * sa], [sx * sa, sy * ca]], np.float32)
    # Transform around the keyboard centre so the result stays mostly inside 0..1
    c = np.array([0.5, 0.5], np.float32)
    shift = np.array([rng.uniform(-cfg.shift, cfg.shift), rng.uniform(-cfg.shift, cfg.shift)], np.float32)

    pts = feats.T - c                              # [64, 2]
    pts = pts @ m.T + c + shift
    k = (keys - c) @ m.T + c + shift
    k = k + np.random.normal(0, cfg.key_jitter, k.shape).astype(np.float32)
    pts = pts + np.random.normal(0, cfg.point_noise, pts.shape).astype(np.float32)
    if cfg.wobble > 0:
        # Low-frequency deviation: a few random sine components along the path
        tt = np.linspace(0, 1, pts.shape[0], dtype=np.float32)[:, None]
        for _ in range(2):
            amp = np.random.normal(0, cfg.wobble / 2, (1, 2)).astype(np.float32)
            f = rng.uniform(0.5, 2.5)
            ph = rng.uniform(0, 2 * math.pi)
            pts = pts + amp * np.sin(2 * math.pi * f * tt + ph)
    pts = np.clip(pts, 0, 1)  # the engine clamps to 0..1
    return pts.T.astype(np.float32), k.astype(np.float32)


def pad_keys(keys: np.ndarray):
    out = np.zeros((MAX_KEYS, 2), np.float32)
    mask = np.zeros(MAX_KEYS, bool)
    n = min(len(keys), MAX_KEYS)
    out[:n] = keys[:n]
    mask[:n] = True
    return out, mask


def targets_for(word: str, letters: str):
    return [letters.index(c) for c in word]
