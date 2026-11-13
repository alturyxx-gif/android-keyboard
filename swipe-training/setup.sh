#!/usr/bin/env bash
# One-time setup on your Linux PC (NVIDIA GPU). Safe to run again.
#
#   cd swipe-training && ./setup.sh
#   PYTHON=/path/to/python3.13 ./setup.sh     # use a specific Python (needs 3.10-3.13)
#
# Creates .venv/ with PyTorch (CUDA) + ExecuTorch 1.2 (the version in the keyboard's swipe
# engine), and builds FUTO's swipe engine in swipe-library/ for the benchmark (evaluate.py).
# Disk: about 15 GB including the dataset later. Time: 10-40 minutes.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
if command -v pacman >/dev/null; then
  PKG_HINT="sudo pacman -S --needed base-devel git cmake python"
else
  PKG_HINT="sudo apt install python3-venv python3-dev git cmake build-essential"
fi

echo "== Checking the system"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  || echo "WARNING: nvidia-smi not found - training will run on the CPU (very slow)."
for tool in "$PY" git cmake g++ make; do
  command -v "$tool" >/dev/null || { echo "Missing '$tool'. Install with: $PKG_HINT"; exit 1; }
done
"$PY" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,13)' 2>/dev/null || {
  echo "$("$PY" --version) can't be used: ExecuTorch 1.2 only has packages for Python 3.10-3.13."
  echo "Get Python 3.13 alongside the system one, e.g. with uv:"
  echo "  sudo pacman -S uv   (or: curl -LsSf https://astral.sh/uv/install.sh | sh)"
  echo "  uv python install 3.13"
  echo "  PYTHON=\$(uv python find 3.13) ./setup.sh"
  exit 1
}
"$PY" -c 'import ensurepip' 2>/dev/null || { echo "Python's venv module is missing. Install with: $PKG_HINT"; exit 1; }
echo "Using $("$PY" --version) at $(command -v "$PY")"

echo "== Python environment (.venv)"
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install -q --upgrade pip
# PyTorch 2.11 with CUDA 12.6 (supports the GTX 16xx series), then ExecuTorch 1.2 for export
.venv/bin/pip install -q "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu126
.venv/bin/pip install -q "torchao==0.18.0" --index-url https://download.pytorch.org/whl/cu126 || .venv/bin/pip install -q "torchao>=0.17.0"
.venv/bin/pip install -q "executorch==1.2.0" numpy
.venv/bin/python - <<'EOF'
import torch, executorch
print("PyTorch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: PyTorch can't see the GPU. Check 'nvidia-smi' works and the driver supports CUDA 12.6.")
EOF

if [ "${1:-}" != "--no-engine" ]; then
  echo "== FUTO's swipe engine (for evaluate.py; takes 10-30 minutes the first time)"
  if [ ! -d swipe-library ]; then
    git clone --recursive https://gitlab.futo.org/keyboard/swipe-library.git
  fi
  # FUTO's Makefile calls 'python3': make it the same Python as above
  SHIM="$(pwd)/.python-shim"; mkdir -p "$SHIM"; ln -sf "$(command -v "$PY")" "$SHIM/python3"
  (cd swipe-library && git submodule update --init --recursive && PATH="$SHIM:$PATH" make host -j"$(nproc)")
  test -x swipe-library/build/infer_jsonl && echo "Engine built: swipe-library/build/infer_jsonl"
fi

echo
echo "Setup done. Next:"
echo "  source .venv/bin/activate"
echo "  python prepare_data.py          # downloads ~6 GB, then prepares it (a few minutes)"
echo "  python train.py --size medium   # overnight"
