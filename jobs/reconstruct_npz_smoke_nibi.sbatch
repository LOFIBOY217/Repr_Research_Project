#!/bin/bash
#SBATCH --job-name=recon-npz-smoke
#SBATCH --account=def-kdhkdh
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

: "${BATCH_SIZE:=2}"
: "${MAX_IMAGES:=16}"

module load StdEnv/2023
module load python/3.11
module load gcc
module load opencv/4.14.0

source ~/projects/def-kdhkdh/jiaqi217/envs/fdloss/bin/activate

export project=~/projects/def-kdhkdh/jiaqi217
export REPR_REPO=$project/Repr_Research_Project
export HF_HOME=$SCRATCH/huggingface
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

INPUT_NPZ="$project/data/eval_refs/VIRTUAL_imagenet256_labeled.npz"
OUTPUT_NPZ="$project/data/eval_refs/VIRTUAL_imagenet256_labeled_reconstructed_sdvae_smoke${MAX_IMAGES}.npz"

cd "$REPR_REPO"
mkdir -p logs

python - <<'PY'
import torch

print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY

python reconstruction/reconstruct_npz.py \
  --input-npz "$INPUT_NPZ" \
  --output-npz "$OUTPUT_NPZ" \
  --batch-size "$BATCH_SIZE" \
  --max-images "$MAX_IMAGES"

OUTPUT_NPZ="$OUTPUT_NPZ" python - <<'PY'
import os
import numpy as np

path = os.environ["OUTPUT_NPZ"]
data = np.load(path)
print("output:", path)
print("files:", data.files)
arr = data["arr_0"]
print("arr_0 shape:", arr.shape)
print("arr_0 dtype:", arr.dtype)
print("arr_0 min/max:", arr.min(), arr.max())
PY
