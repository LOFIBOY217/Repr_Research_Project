#!/bin/bash
#SBATCH --job-name=recon-npz
#SBATCH --account=def-kdhkdh
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

: "${BATCH_SIZE:=4}"
: "${MAX_IMAGES:=}"

module load StdEnv/2023
module load python/3.11
module load gcc
module load opencv/4.14.0

source ~/projects/def-kdhkdh/jiaqi217/envs/fdloss/bin/activate

export project=~/projects/def-kdhkdh/jiaqi217
export REPR_REPO=$project/Repr_Research_Project
export HF_HOME=$SCRATCH/huggingface
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$REPR_REPO"
mkdir -p logs

python - <<'PY'
import torch

print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY

cmd=(python reconstruction/reconstruct_npz.py --batch-size "$BATCH_SIZE")
if [[ -n "$MAX_IMAGES" ]]; then
  cmd+=(--max-images "$MAX_IMAGES")
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"

python - <<'PY'
import numpy as np
from pathlib import Path

path = Path.home() / "projects/def-kdhkdh/jiaqi217/data/eval_refs/VIRTUAL_imagenet256_labeled_reconstructed_sdvae.npz"
data = np.load(path)
print("output:", path)
print("files:", data.files)
arr = data["arr_0"]
print("arr_0 shape:", arr.shape)
print("arr_0 dtype:", arr.dtype)
print("arr_0 min/max:", arr.min(), arr.max())
PY
