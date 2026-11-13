#!/usr/bin/env python3
"""
Turn a trained checkpoint into the files the keyboard imports.

    python export.py runs/medium/best.pt                 # -> exports/medium/
    python export.py runs/personal/best.pt --name mine   # -> exports/mine/

Writes model_fp32.pte (ExecuTorch 1.2, the version in the keyboard's swipe engine) and
metadata.json, after checking that the model only uses operations the phone supports and
gives the same results as in training. Then run evaluate.py on the export folder.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from swipekit.export_pte import export, make_zip, write_metadata  # noqa: E402
from swipekit.model import count_params  # noqa: E402
from swipekit.training import load_model  # noqa: E402

# Search settings in the phone's engine are built in and chosen by model name. A custom model
# starts with FUTO's settings for their encoder; evaluate.py --tune picks the best preset.
DEFAULT_CODENAME = "honorable_sturgeon"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--name", default=None, help="export folder name (default: run folder name)")
    ap.add_argument("--out", default="exports")
    ap.add_argument("--codename", default=DEFAULT_CODENAME)
    args = ap.parse_args()

    ck_path = Path(args.checkpoint)
    name = args.name or ck_path.parent.name
    out = Path(args.out) / name
    model = load_model(ck_path)
    print(f"Exporting {ck_path} ({count_params(model) / 1e6:.2f}M parameters) -> {out}/")
    info = export(model, out / "model_fp32.pte")
    import torch
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    meta = write_metadata(out, "model_fp32.pte", args.codename, model.cfg.to_dict(), {
        "display_name": f"custom-{name}",
        "training": {"epoch": ck.get("epoch"), "dev_loss": ck.get("dev_loss"), "note": ck.get("note", "")},
        "undelegated_ops": info["undelegated_ops"],
    })
    print(f"OK: phone-compatible operations only; matches PyTorch (max difference {info['max_abs_diff']:.1e})")
    print(f"     {out / 'model_fp32.pte'} ({meta['file_sizes']['model_fp32.pte'] / 1e6:.1f} MB)")
    z = make_zip(out)
    print(f"     {z}  <- the file to import on the phone (after evaluate.py --tune)")
    print(f"Next: python evaluate.py {out} --tune")


if __name__ == "__main__":
    main()
