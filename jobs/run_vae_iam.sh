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

# Default preview: 3 samples. Use MAX_SAMPLES=all for full-set metrics.
args=(--config "${CONFIG:-configs/dit_val_iam.yaml}")
if [ -n "${CHECKPOINT:-}" ]; then args+=(--checkpoint "$CHECKPOINT"); fi
if [ -n "${DATA_FILE:-}" ]; then args+=(--input "$DATA_FILE"); fi
if [ -n "${SAMPLE_KEYS:-}" ]; then
    read -r -a sample_keys <<< "$SAMPLE_KEYS"
    args+=(--sample-keys "${sample_keys[@]}")
    # Explicit selection includes every requested sample unless a limit is supplied.
    sample_limit="${MAX_SAMPLES:-all}"
else
    sample_limit="${MAX_SAMPLES:-3}"
fi
if [ "$sample_limit" != "all" ]; then args+=(--max-samples "$sample_limit"); fi
if [ "${METRICS_ONLY:-0}" = "1" ]; then args+=(--metrics-only); fi
python -u val_vae_iam.py "${args[@]}"
