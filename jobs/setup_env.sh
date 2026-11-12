#!/bin/bash
# Source from a job script after setting DIFFINK_ROOT.
module load Anaconda3/2022.05

export PATH="/storage/homefs/mj24z011/.conda/envs/diffink/bin:$PATH"
export NVRTC_DIR="/storage/homefs/mj24z011/.conda/envs/diffink/lib/python3.8/site-packages/nvidia/cuda_nvrtc/lib"

export LD_LIBRARY_PATH="$NVRTC_DIR:${LD_LIBRARY_PATH:-}"
echo "Checking NVRTC library:"
ls -lh "$NVRTC_DIR"

python - <<'PY'
import ctypes
ctypes.CDLL("libnvrtc.so")
print("NVRTC loaded successfully!")
PY

echo "Python path:"
which python

echo "Checking environment:"
python -c "import sys, torch, wandb; print(sys.executable); print(torch.__version__); print(wandb.__version__)"

cd "$DIFFINK_ROOT"
