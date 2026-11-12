#!/bin/bash
#SBATCH --job-name=vae_iam_train
#SBATCH --partition=gpu
#SBATCH --qos=job_gratis
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=vae_iam_train_%j.out
#SBATCH --error=vae_iam_train_%j.err

set -euo pipefail

# Slurm copies submitted scripts; use the project path instead of script dirname.
export DIFFINK_ROOT="${DIFFINK_ROOT:-/storage/homefs/mj24z011/DiffInk-IAM}"
source "$DIFFINK_ROOT/jobs/setup_env.sh"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# No RESUME: initialize all VAE/OCR/style parameters from scratch.
if [ -n "${RESUME:-}" ]; then
    python -u train_vae_iam.py --config configs/vae_iam_scratch.yaml --resume "$RESUME"
else
    python -u train_vae_iam.py --config configs/vae_iam_scratch.yaml
fi
