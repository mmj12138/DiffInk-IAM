#!/bin/bash
#SBATCH --job-name=vae_iam_check
#SBATCH --partition=gpu
#SBATCH --qos=job_gratis
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=vae_iam_check_%j.out
#SBATCH --error=vae_iam_check_%j.err

set -euo pipefail

# Slurm copies submitted scripts; use the project path instead of script dirname.
export DIFFINK_ROOT="${DIFFINK_ROOT:-/storage/homefs/mj24z011/DiffInk-IAM}"
source "$DIFFINK_ROOT/jobs/setup_env.sh"

# Default: inspect 3 samples. Set MAX_SAMPLES=all to process all 87.
if [ "${MAX_SAMPLES:-3}" = "all" ]; then
    python -u val_vae_iam.py --config configs/dit_val_iam.yaml
else
    python -u val_vae_iam.py --config configs/dit_val_iam.yaml --max-samples "${MAX_SAMPLES:-3}"
fi

