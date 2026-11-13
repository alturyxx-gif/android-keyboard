"""
Export a trained encoder to an ExecuTorch 1.2 `.pte` file that FUTO's swipe engine (and so the
keyboard) can load, and check it before you copy it to the phone.

Checks:
  1. Every operation that isn't handled by XNNPACK is in the phone engine's op list
     (the engine is built with only the ops FUTO's own models use).
  2. The exported file gives the same answers as the PyTorch model.
"""
import json
import hashlib
from pathlib import Path

import torch

from .model import SwipeEncoder, T_IN, MAX_KEYS

# Union of FUTO's models/*/ops.txt in swipe-library (honorable_sturgeon, hungry_jellyfish,
# magic_macaw): the only non-XNNPACK kernels compiled into the phone's swipe engine.
PHONE_ALLOWED_OPS = {
    "aten::_log_softmax.out", "aten::arange.start_out", "aten::atan2.out", "aten::bitwise_not.out",
    "aten::cumsum.out", "aten::expand_copy.out", "aten::full_like.out", "aten::gt.Scalar_out",
    "aten::lt.Scalar_out", "aten::scalar_tensor.out", "aten::select_copy.int_out",
    "aten::split_with_sizes_copy.out", "aten::squeeze_copy.dims_out", "aten::unsqueeze_copy.out",
    "aten::where.self_out", "dim_order_ops::_to_dim_order_copy.out", "aten::clamp.out",
    "aten::embedding.out", "aten::native_layer_norm.out", "aten::sum.IntList_out",
}


def example_inputs(batch: int = 1):
    feats = torch.rand(batch, 2, T_IN)
    keys = torch.zeros(batch, MAX_KEYS, 2)
    mask = torch.zeros(batch, MAX_KEYS, dtype=torch.bool)
    keys[:, :26] = torch.rand(batch, 26, 2)
    mask[:, :26] = True
    return feats, keys, mask


def _undelegated_ops(et_program) -> set:
    """Operator names left for the portable kernels (i.e. not inside an XNNPACK delegate)."""
    ops = set()
    prog = et_program.executorch_program
    for plan in prog.execution_plan:
        for op in plan.operators:
            name = op.name + ("." + op.overload if op.overload else "")
            ops.add(name)
    return ops


def _ensure_flatc_on_path():
    """ExecuTorch's pip package ships `flatc` but doesn't put it on PATH."""
    import os
    import executorch
    bin_dir = Path(list(executorch.__path__)[0]) / "data" / "bin"
    if (bin_dir / "flatc").is_file() and str(bin_dir) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def export(model: SwipeEncoder, out_path: Path) -> dict:
    _ensure_flatc_on_path()
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower

    model = model.eval().cpu()
    inputs = example_inputs()
    with torch.no_grad():
        exported = torch.export.export(model, inputs)
        edge = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
        et_program = edge.to_executorch()

    ops = _undelegated_ops(et_program)
    bad = sorted(o for o in ops if o not in PHONE_ALLOWED_OPS)
    if bad:
        raise RuntimeError(
            "The model uses operations the phone's swipe engine doesn't include: " + ", ".join(bad))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(et_program.buffer)

    # Compare the exported program with PyTorch on random inputs
    from executorch.runtime import Runtime
    program = Runtime.get().load_program(str(out_path))  # keep a reference while executing
    method = program.load_method("forward")
    max_diff = 0.0
    for _ in range(5):
        f, k, m = example_inputs()
        with torch.no_grad():
            ref = model(f, k, m)[0]
        got = method.execute((f, k, m))[0]
        valid = m[0].nonzero().flatten().tolist() + [MAX_KEYS]
        max_diff = max(max_diff, float((ref[..., valid] - got[..., valid]).abs().max()))
    if max_diff > 1e-3:
        raise RuntimeError(f"Exported model differs from PyTorch (max difference {max_diff:.5f})")

    return {"undelegated_ops": sorted(ops), "max_abs_diff": max_diff}


def write_metadata(out_dir: Path, pte_name: str, codename: str, cfg: dict, extra: dict):
    """metadata.json next to the .pte, in the format FUTO's engine reads."""
    pte = out_dir / pte_name
    meta = {
        "kind": "encoder",
        "codename": codename,
        "layout": "any",
        "input_shape": [1, 2, T_IN],
        "input_shapes": {"features": [1, 2, T_IN], "layout_keys": [1, MAX_KEYS, 2], "layout_mask": [1, MAX_KEYS]},
        "max_keys": MAX_KEYS,
        # FUTO's engine reads the coefficient size as freq_x * freq_y (= COEFF_DIM = 64)
        "model_config": {**cfg, "freq_x": 8, "freq_y": 8},
        "file_hashes": {pte_name: hashlib.sha256(pte.read_bytes()).hexdigest()},
        "file_sizes": {pte_name: pte.stat().st_size},
        "custom": True,
        **extra,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def make_zip(model_dir: Path) -> Path:
    """The single file you copy to the phone (Swipe model & training > Import model)."""
    import zipfile
    model_dir = Path(model_dir)
    z = model_dir / f"swipe-model-{model_dir.name}.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
        f.write(model_dir / "model_fp32.pte", "model_fp32.pte")
        f.write(model_dir / "metadata.json", "metadata.json")
    return z
