#!/usr/bin/env bash
# One CUDA GPU; engineering checks only. Account and GPU supplied by sbatch.
#SBATCH --job-name=recon-C-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:45:00
#SBATCH --output=recon-C-smoke-%j.out
set -euo pipefail
: "${FD_PROJECT:?Set absolute project directory}"
: "${FD_PYTHON:?Set existing Python executable}"
: "${FD_C_SMOKE_BASE:?Set output parent directory}"
: "${FD_EXPECTED_COMMIT:?Set the reviewed Git commit}"
: "${IMAGENET_ROOT:?Set ImageNet train/val root}"
: "${TOKENIZER_CHECKPOINT:?Set local AutoencoderKL checkpoint}"
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
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
export FD_C_SMOKE_ROOT="$FD_C_SMOKE_BASE/job-$SLURM_JOB_ID"
cd "$FD_PROJECT"
test "$(git rev-parse HEAD)" = "$FD_EXPECTED_COMMIT"
git diff --quiet
git diff --cached --quiet
nvidia-smi
exec "$FD_PYTHON" scripts/ours_nibi_smoke.py
