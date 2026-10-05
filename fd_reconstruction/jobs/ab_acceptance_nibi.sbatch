#!/usr/bin/env bash
# Exactly one of A or B on ONE GPU. Submit independent groups one at a time.
# Supply account and --gpus=h100:1 when submitting; no forced partition.
#SBATCH --job-name=recon-single-accept
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=recon-AB-accept-%j.out
set -euo pipefail
: "${FD_PROJECT:?Set an isolated project checkout}"
: "${FD_PYTHON:?Set existing cluster Python executable}"
: "${FD_AB_SMOKE_BASE:?Set output parent directory}"
: "${FD_EXPECTED_COMMIT:?Set reviewed Git commit}"
: "${FD_ACCEPT_GROUP:?Choose exactly A or B}"
case "$FD_ACCEPT_GROUP" in A|B) ;; *) echo "FD_ACCEPT_GROUP must be A or B" >&2; exit 2 ;; esac
: "${IMAGENET_ROOT:?Set ImageNet root}"
: "${TOKENIZER_CHECKPOINT:?Set local AutoencoderKL checkpoint}"
: "${INCEPTION_WEIGHTS:?Set local Inception checkpoint}"
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
export FD_AB_SMOKE_ROOT="$FD_AB_SMOKE_BASE/job-$SLURM_JOB_ID"
cd "$FD_PROJECT"
test "$(git rev-parse HEAD)" = "$FD_EXPECTED_COMMIT"
git diff --quiet
git diff --cached --quiet
nvidia-smi
exec "$FD_PYTHON" scripts/ab_acceptance.py
