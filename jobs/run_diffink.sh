#!/bin/bash
#SBATCH --job-name=diffink_test
#SBATCH --partition=gpu
#SBATCH --qos=job_gratis
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=diffink_%j.out
#SBATCH --error=diffink_%j.err

set -euo pipefail

# Slurm copies submitted scripts; use the project path instead of script dirname.
export DIFFINK_ROOT="${DIFFINK_ROOT:-/storage/homefs/mj24z011/DiffInk-IAM}"
source "$DIFFINK_ROOT/jobs/setup_env.sh"

python val_dit.py

