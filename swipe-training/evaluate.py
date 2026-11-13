#!/usr/bin/env python3
"""
Compare your model with FUTO's, using FUTO's real swipe engine (the same code the keyboard
runs) and the same English dictionary as the keyboard.

    python evaluate.py exports/medium --tune            # pick search settings, then compare
    python evaluate.py exports/medium                   # compare only
    python evaluate.py exports/mine --personal futo-swipes-*.jsonl.gz

Prints the share of swipes where the right word is the first guess ("top-1") and among the
four suggestions ("top-4"), on FUTO's test swipes (other people, never trained on) and, with
--personal, on the newest 15% of your own swipes (never trained on).

Decision rule (SWIPE_PLAN.md): use your model only if it beats FUTO's full setup on both.
"""
import argparse
import warnings
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

# PyTorch/ExecuTorch print many harmless FutureWarnings during export
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).parent))
from swipekit import engine as E  # noqa: E402
from swipekit.data import to_engine_jsonl  # noqa: E402
from swipekit.export_pte import make_zip  # noqa: E402
from swipekit.personal import load_lines, split  # noqa: E402

PRESETS = ["honorable_sturgeon", "custom"]  # FUTO's tuned settings / the engine's fallback settings


def pct(r):
    return f"{r['top1'] * 100:5.1f}%   {r['top4'] * 100:5.1f}%   {r['ms']:6.1f} ms" if r else "   -"


def personal_groups(lines, tmp: Path):
    """Group personal test swipes by keyboard shape -> (layout.json, swipes.jsonl, refiner?)."""
    groups = defaultdict(list)
    for line in lines:
        d = json.loads(line)
        fr = d.get("frame")
        if not fr:
            continue
        key = (fr["letters"], tuple(round(v, 3) for v in fr["xs"]), tuple(round(v, 3) for v in fr["ys"]),
               bool(fr.get("refiner")))
        conv = to_engine_jsonl(line)
        if conv:
            groups[key].append(conv)
    out = []
    for i, (key, rows) in enumerate(groups.items()):
        letters, xs, ys, refiner = key
        lay = E.layout_json(letters, xs, ys, tmp / f"layout{i}.json")
        jl = tmp / f"personal{i}.jsonl"
        jl.write_text("\n".join(rows) + "\n")
        out.append((lay, jl, refiner, len(rows)))
    return out


def combine(results):
    n = sum(r["n"] for r in results)
    if not n:
        return None
    return {"n": n,
            "top1": sum(r["top1"] * r["n"] for r in results) / n,
            "top4": sum(r["top4"] * r["n"] for r in results) / n,
            "ms": sum((r["ms"] or 0) * r["n"] for r in results) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", help="export folder (model_fp32.pte + metadata.json)")
    ap.add_argument("--tune", action="store_true", help="pick the best search-settings preset on dev swipes")
    ap.add_argument("--personal", nargs="*", default=[], help="your exported swipe files")
    ap.add_argument("--data", default="data")
    ap.add_argument("--n", type=int, default=10000, help="test swipes to use")
    ap.add_argument("--engine", default="swipe-library/build")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--skip-futo", action="store_true")
    args = ap.parse_args()

    cache = Path(args.cache)
    models = E.futo_models(cache / "futo")
    vocab = E.vocab(cache)
    eng = E.Engine(Path(args.engine), models, vocab, Path(args.engine).parent / "models" / "layouts")
    mdir = Path(args.model_dir)
    tmp = Path(tempfile.mkdtemp(prefix="swipe_eval_"))
    data = Path(args.data)

    if args.tune:
        dev = E.head(data / "dev.jsonl", 5000, tmp / "dev.jsonl")
        print("Choosing search settings on 5000 dev swipes...")
        best = None
        for preset in PRESETS:
            r = eng.run(E.with_codename(mdir, preset), dev, lm=True)
            print(f"  preset {preset:20s} top-1 {r['top1'] * 100:.2f}%  top-4 {r['top4'] * 100:.2f}%")
            if best is None or r["top1"] > best[1]["top1"]:
                best = (preset, r)
        meta = json.loads((mdir / "metadata.json").read_text())
        meta["codename"] = best[0]
        (mdir / "metadata.json").write_text(json.dumps(meta, indent=2))
        z = make_zip(mdir)
        print(f"-> using '{best[0]}' (saved in {mdir / 'metadata.json'}; phone file updated: {z})\n")

    test = E.head(data / "test.jsonl", args.n, tmp / "test.jsonl")
    rows = []
    print(f"FUTO test swipes (first {args.n}, other people's swiping)...")
    if not args.skip_futo:
        fe = models / "honorable_sturgeon/model_fp32.pte"
        rows.append(("FUTO: model + refiner + context", eng.run(fe, test, decoder=True, lm=True)))
        rows.append(("FUTO: model + context (no refiner)", eng.run(fe, test, decoder=False, lm=True)))
    rows.append((f"Yours ({mdir.name}) + context", eng.run(mdir / "model_fp32.pte", test, lm=True)))

    personal_rows = []
    if args.personal:
        lines = load_lines(args.personal)
        _, _, ptest = split(lines)
        print(f"Your swipes: {len(lines)} total, testing on the newest {len(ptest)}...")
        groups = personal_groups(ptest, tmp)
        if groups:
            futo = []
            mine = []
            for lay, jl, refiner, n in groups:
                if not args.skip_futo:
                    futo.append(eng.run(models / "honorable_sturgeon/model_fp32.pte", jl,
                                        decoder=refiner, lm=True, layout=lay))
                mine.append(eng.run(mdir / "model_fp32.pte", jl, lm=True, layout=lay))
            if futo:
                personal_rows.append(("FUTO (as on your phone)", combine(futo)))
            personal_rows.append((f"Yours ({mdir.name})", combine(mine)))

    print("\n                                          top-1    top-4    time/swipe (PC)")
    print("FUTO test swipes:")
    for name, r in rows:
        print(f"  {name:38s} {pct(r)}")
    if personal_rows:
        print("Your own swipes (newest 15%, never trained on):")
        for name, r in personal_rows:
            print(f"  {name:38s} {pct(r)}")

    if not args.skip_futo:
        futo_best = rows[0][1]["top1"]
        mine = rows[-1][1]["top1"]
        verdict = "BEATS" if mine > futo_best else "does NOT beat"
        print(f"\nOn FUTO's test swipes your model {verdict} FUTO's full setup "
              f"({mine * 100:.1f}% vs {futo_best * 100:.1f}% first guess).")
        if personal_rows and len(personal_rows) == 2:
            pf, pm = personal_rows[0][1]["top1"], personal_rows[1][1]["top1"]
            print(f"On your own swipes: {pm * 100:.1f}% vs FUTO {pf * 100:.1f}% "
                  f"({'better' if pm > pf else 'not better'}).")
    (mdir / "evaluation.json").write_text(json.dumps({"futo_test": rows, "personal": personal_rows}, indent=2))


if __name__ == "__main__":
    main()
