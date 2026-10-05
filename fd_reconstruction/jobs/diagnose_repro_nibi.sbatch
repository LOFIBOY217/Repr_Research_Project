#!/usr/bin/env bash
# Isolated diagnosis only; production algorithm and tolerance unchanged.
#SBATCH --job-name=recon-repro-diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:20:00
#SBATCH --output=recon-repro-diag-%j.out
set -euo pipefail
: "${FD_PROJECT:?Set clean checkout}"
: "${FD_PYTHON:?Set Python}"
: "${FD_EXPECTED_COMMIT:?Set reviewed commit}"
: "${FD_DIAG_BASE:?Set new output parent}"
: "${FD_AB_SOURCE:?Set completed A/B run root}"
: "${FD_C_SOURCE:?Set completed current-both C run root}"
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
export FD_DIAG_ROOT="$FD_DIAG_BASE/job-$SLURM_JOB_ID"
cd "$FD_PROJECT"
test "$(git rev-parse HEAD)" = "$FD_EXPECTED_COMMIT"
git diff --quiet
git diff --cached --quiet
nvidia-smi
exec "$FD_PYTHON" scripts/diagnose_repro.py
