"""Tests for the training kit:  python -m pytest tests  (from swipe-training/)."""
import gzip
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from swipekit import data as D  # noqa: E402
from swipekit.model import SwipeEncoder, ModelConfig, T_OUT, MAX_KEYS  # noqa: E402
from swipekit.personal import load_lines, split  # noqa: E402
from swipekit.training import SwipeSet, collate, ctc_loss, fits_ctc  # noqa: E402


def straight_path(word, step_ms=16.0, per_key=6):
    pts = []
    t = 0.0
    for a, b in zip(word, word[1:]):
        (x0, y0), (x1, y1) = D.QWERTY[a], D.QWERTY[b]
        for i in range(per_key):
            f = i / per_key
            pts.append((x0 + f * (x1 - x0), y0 + f * (y1 - y0), t))
            t += step_ms
    x1, y1 = D.QWERTY[word[-1]]
    pts.append((x1, y1, t))
    return pts


def test_engine_features_shape_and_ends():
    p = straight_path("hello")
    f = D.engine_features([q[0] for q in p], [q[1] for q in p], [q[2] for q in p])
    assert f.shape == (2, 64)
    assert np.allclose(f[:, 0], D.QWERTY["h"], atol=1e-5)
    assert np.allclose(f[:, -1], D.QWERTY["o"], atol=1e-5)
    assert f.min() >= 0 and f.max() <= 1


def test_resample_60hz_point_count_matches_engine_rule():
    # duration 500 ms -> round(500 / 16.667) + 1 = 31 points
    x, y = D.resample_60hz([0, 1], [0, 1], [0, 500])
    assert len(x) == 31 and abs(x[-1] - 1) < 1e-6


def test_dataset_line_parsing_and_lr_scale():
    p = straight_path("cat")
    line = json.dumps({"word": "Cat", "data": [{"x": a, "y": b, "t": c + 1000} for a, b, c in p],
                       "sentence": "the Cat sat", "word_idx": 1})
    s = D.parse_dataset_line(line)
    assert s.word == "cat" and s.context == ["the"]
    lr = json.dumps({"word": "cat", "data": {"L": [[{"x": a, "y": b / D.LR_Y_SCALE, "t": c} for a, b, c in p]], "R": []}})
    s2 = D.parse_dataset_line(lr)
    assert np.allclose(s.feats, s2.feats, atol=1e-5)
    assert D.parse_dataset_line(json.dumps({"word": "café", "data": []})) is None
    assert D.parse_dataset_line(json.dumps({"word": "cat", "language": "de", "data": []})) is None


def keyboard_record(word, time=1):
    """A record exactly as the keyboard's SwipeRecording exports it (refiner frame)."""
    p = straight_path(word)
    # Keyboard-normalised coordinates: model y = y * 4/3 * sy + oy; pick sy=0.75, oy=0
    return json.dumps({"v": 1, "time": time, "word": word, "source": "accepted",
                       "candidates": [word], "context": ["say"],
                       "path": {"x": [q[0] for q in p], "y": [q[1] for q in p], "t": [q[2] for q in p]},
                       "keys": {"letters": D.LETTERS, "x": [D.QWERTY[c][0] for c in D.LETTERS],
                                "y": [D.QWERTY[c][1] for c in D.LETTERS]},
                       "frame": {"letters": D.LETTERS, "xs": [D.QWERTY[c][0] for c in D.LETTERS],
                                 "ys": [D.QWERTY[c][1] for c in D.LETTERS],
                                 "sx": 1.0, "sy": 0.75, "ox": 0.0, "oy": 0.0,
                                 "refiner": True, "contextModel": True}})


def test_personal_record_uses_phone_frame():
    s = D.parse_personal_line(keyboard_record("hello"))
    assert s.word == "hello" and s.letters == D.LETTERS
    # With sy=0.75 the 4/3 factor cancels: path ends on the keys
    assert np.allclose(s.feats[:, 0], D.QWERTY["h"], atol=1e-4)
    eng = json.loads(D.to_engine_jsonl(keyboard_record("hello")))
    assert eng["word"] == "hello" and eng["word_idx"] == 1 and len(eng["data"]) > 3


def test_personal_split_is_chronological(tmp_path):
    f = tmp_path / "swipes.jsonl.gz"
    with gzip.open(f, "wt") as g:
        for i in reversed(range(100)):
            g.write(keyboard_record("hello", time=i) + "\n")
    lines = load_lines([f])
    tr, va, te = split(lines)
    assert (len(tr), len(va), len(te)) == (75, 10, 15)
    assert json.loads(te[-1])["time"] == 99 and json.loads(tr[0])["time"] == 0


def test_ctc_fit_rule():
    assert fits_ctc("hello")
    assert not fits_ctc("a" * 17)  # 17 letters + 16 repeats > 32
    assert not fits_ctc("x" * 33)


def test_augment_keeps_path_and_keys_together():
    rng = random.Random(1)
    p = straight_path("word")
    f = D.engine_features([q[0] for q in p], [q[1] for q in p], [q[2] for q in p])
    cfg = D.AugmentConfig(p=1.0, point_noise=0, key_jitter=0, wobble=0, shift=0, rotate_deg=0)
    f2, k2 = D.augment(f, D.QWERTY_KEYS, cfg, rng)
    w = D.LETTERS.index("w")
    assert np.allclose(f2[:, 0], k2[w], atol=1e-4)  # still starts on the (moved) w key


def test_model_shapes_and_masking():
    m = SwipeEncoder(ModelConfig.preset("tiny")).eval()
    ds = SwipeSet([D.engine_features(*zip(*straight_path("hello")))] * 2, ["hello", "hello"])
    f, k, mk, tg, tl = collate([ds[0], ds[1]])
    le, coeff, lam = m(f, k, mk)
    assert le.shape == (2, T_OUT, MAX_KEYS + 1) and coeff.shape == (2, T_OUT, 64) and lam.shape == (2, T_OUT, 1)
    probs = le.exp()
    assert torch.allclose(probs.sum(-1), torch.ones(2, T_OUT), atol=1e-4)
    assert probs[..., 26:MAX_KEYS].max() < 1e-3  # padded key slots get no probability
    assert torch.isfinite(ctc_loss(le, tg, tl))


def test_export_is_phone_compatible(tmp_path):
    from swipekit.export_pte import export, make_zip, write_metadata
    m = SwipeEncoder(ModelConfig.preset("tiny"))
    info = export(m, tmp_path / "m" / "model_fp32.pte")  # raises if an op isn't on the phone
    assert info["max_abs_diff"] < 1e-3
    meta = write_metadata(tmp_path / "m", "model_fp32.pte", "honorable_sturgeon", m.cfg.to_dict(), {})
    assert meta["model_config"]["freq_x"] * meta["model_config"]["freq_y"] == 64
    import zipfile
    names = zipfile.ZipFile(make_zip(tmp_path / "m")).namelist()
    assert sorted(names) == ["metadata.json", "model_fp32.pte"]
