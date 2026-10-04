#!/usr/bin/env bash
# Engineering only: no scientific small-sample FD claims.
# Supply --account and a single CUDA GPU type when submitting.
#SBATCH --job-name=recon-fd-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:45:00
#SBATCH --output=recon-fd-smoke-%j.out
set -euo pipefail
: "${FD_PROJECT:?Set absolute project directory}"
: "${FD_PYTHON:?Set existing cluster Python executable}"
: "${FD_SMOKE_ROOT:?Set a new absolute output directory}"
: "${IMAGENET_ROOT:?Set original ImageNet train/val directory}"
: "${TOKENIZER_CHECKPOINT:?Set local pretrained AutoencoderKL directory}"
: "${INCEPTION_WEIGHTS:?Set local official Inception checkpoint}"

module load StdEnv/2023
module load python/3.11
module load gcc
module load opencv/4.14.0

export PYTHONPATH="$FD_PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
cd "$FD_PROJECT"
nvidia-smi
exec "$FD_PYTHON" scripts/nibi_smoke.py
