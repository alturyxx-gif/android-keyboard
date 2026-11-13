"""Runs swipes through FUTO's real swipe engine (swipe-library's infer_jsonl tool)."""
import json
import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path

FUTO_REPO = "https://huggingface.co/futo-org/futo-swipe/resolve/prod/"
FUTO_FILES = [
    "honorable_sturgeon/model_fp32.pte", "honorable_sturgeon/metadata.json",
    "magic_macaw/model_fp32.pte", "magic_macaw/metadata.json",
    "chosen_planarian/context_lm.pte", "chosen_planarian/metadata.json", "chosen_planarian/vocab.txt",
]
VOCAB_URL = "https://dl.keyboard.futo.org/en_w2.combined"  # same English dictionary as the keyboard


def fetch(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    # Some servers reject Python's default client name
    req = urllib.request.Request(url, headers={"User-Agent": "swipe-training-kit/1.0"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return dest


def futo_models(root: Path) -> Path:
    for f in FUTO_FILES:
        fetch(FUTO_REPO + f, root / f)
    return root


def vocab(root: Path) -> Path:
    return fetch(VOCAB_URL, root / "en_w2.combined")


def layout_json(letters, xs, ys, path: Path):
    keys = [{"letter": c, "cx": float(x), "cy": float(y), "rx": 0.05, "ry": 0.1667}
            for c, x, y in zip(letters, xs, ys)]
    path.write_text(json.dumps({"name": "custom", "letters": letters, "keys": keys}))
    return path


class Engine:
    def __init__(self, build_dir: Path, models_dir: Path, vocab_path: Path, layouts_dir: Path):
        self.exe = Path(build_dir) / "infer_jsonl"
        if not self.exe.exists():
            raise FileNotFoundError(f"{self.exe} not found - run setup.sh (it builds FUTO's swipe engine)")
        self.models = Path(models_dir)
        self.vocab = Path(vocab_path)
        self.qwerty = Path(layouts_dir) / "en_qwerty.json"

    def run(self, encoder: Path, jsonl: Path, *, decoder=False, lm=True, layout: Path = None,
            beam=100, threads=4):
        cmd = [str(self.exe), "--encoder", str(encoder), "--vocab", str(self.vocab),
               "--layout", str(layout or self.qwerty), "--jsonl", str(jsonl),
               "--threads", str(threads), "--beam-width", str(beam), "--top-k", "4"]
        if decoder:
            cmd += ["--decoder", str(self.models / "magic_macaw/model_fp32.pte")]
        if lm:
            cmd += ["--lm-model", str(self.models / "chosen_planarian/context_lm.pte"),
                    "--lm-vocab", str(self.models / "chosen_planarian/vocab.txt")]
        out = subprocess.run(cmd, capture_output=True, text=True)
        text = out.stdout + out.stderr
        m1 = re.search(r"top-1:\s+(\d+) / (\d+)", text)
        m4 = re.search(r"top-4:\s+(\d+) / (\d+)", text)
        mt = re.search(r"total:\s+([\d.]+)", text)
        if not m1:
            raise RuntimeError("Engine run failed:\n" + text[-2000:])
        n = int(m1.group(2))
        return {"n": n, "top1": int(m1.group(1)) / n, "top4": int(m4.group(1)) / n,
                "ms": float(mt.group(1)) / 1000 if mt else None}


def head(src: Path, n: int, dest: Path) -> Path:
    with open(src) as f, open(dest, "w") as g:
        for _, l in zip(range(n), f):
            g.write(l)
    return dest


def with_codename(model_dir: Path, codename: str) -> Path:
    """Copy of the export folder whose metadata uses `codename` (search settings preset)."""
    tmp = Path(tempfile.mkdtemp(prefix="swipe_preset_"))
    (tmp / "model_fp32.pte").symlink_to((model_dir / "model_fp32.pte").resolve())
    meta = json.loads((model_dir / "metadata.json").read_text())
    meta["codename"] = codename
    (tmp / "metadata.json").write_text(json.dumps(meta))
    return tmp / "model_fp32.pte"
