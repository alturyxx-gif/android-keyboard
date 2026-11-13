"""Your exported swipes (Settings > Swipe > Swipe model & training > Export)."""
import gzip
import json
from pathlib import Path

TEST_FRACTION = 0.15   # newest 15%: never trained on, used to compare models
VAL_FRACTION = 0.10    # next newest 10%: picks the best fine-tuning epoch


def load_lines(paths) -> list:
    """All records from one or more export files, oldest first, duplicates removed."""
    seen, out = set(), []
    for p in paths:
        p = Path(p)
        opener = gzip.open if p.suffix == ".gz" else open
        with opener(p, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                key = (d.get("time"), d.get("word"))
                if key in seen:
                    continue
                seen.add(key)
                out.append((d.get("time", 0), line))
    out.sort(key=lambda x: x[0])
    return [l for _, l in out]


def split(lines: list):
    """Chronological split: train (oldest 75%), val, test (newest 15%)."""
    n = len(lines)
    n_test = int(n * TEST_FRACTION)
    n_val = int(n * VAL_FRACTION)
    train = lines[: n - n_test - n_val]
    val = lines[n - n_test - n_val: n - n_test]
    test = lines[n - n_test:]
    return train, val, test
