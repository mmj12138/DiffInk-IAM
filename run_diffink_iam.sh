#!/bin/bash
#SBATCH --job-name=diffink_iam
#SBATCH --partition=gpu
#SBATCH --qos=job_gratis
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=diffink_iam_%j.out
#SBATCH --error=diffink_iam_%j.err

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

cd /storage/homefs/mj24z011/DiffInk-IAM

python val_dit_iam.py --config configs/dit_val_iam.yaml

