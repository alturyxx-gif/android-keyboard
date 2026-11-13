#!/usr/bin/env bash
# One-time setup on your Linux PC (NVIDIA GPU). Safe to run again.
#
#   cd swipe-training && ./setup.sh
#
# Creates .venv/ with PyTorch (CUDA) + ExecuTorch 1.2 (the version in the keyboard's swipe
# engine), and builds FUTO's swipe engine in swipe-library/ for the benchmark (evaluate.py).
# Disk: about 15 GB including the dataset later. Time: 10-40 minutes.
set -euo pipefail
cd "$(dirname "$0")"

echo "== Checking the system"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  || echo "WARNING: nvidia-smi not found - training will run on the CPU (very slow)."
for tool in python3 git cmake g++ make; do
  command -v "$tool" >/dev/null || { echo "Missing '$tool'. On Ubuntu: sudo apt install python3-venv python3-dev git cmake build-essential"; exit 1; }
done
python3 -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,13), "Python 3.10-3.13 needed"' \
  || { echo "Python 3.10-3.13 is needed (ExecuTorch 1.2 has no wheels for other versions)."; exit 1; }
python3 -c 'import ensurepip' 2>/dev/null || { echo "Run: sudo apt install python3-venv"; exit 1; }

echo "== Python environment (.venv)"
[ -d .venv ] || python3 -m venv .venv
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
  (cd swipe-library && git submodule update --init --recursive && make host -j"$(nproc)")
  test -x swipe-library/build/infer_jsonl && echo "Engine built: swipe-library/build/infer_jsonl"
fi

echo
echo "Setup done. Next:"
echo "  source .venv/bin/activate"
echo "  python prepare_data.py          # downloads ~6 GB, then prepares it (a few minutes)"
echo "  python train.py --size medium   # overnight"
