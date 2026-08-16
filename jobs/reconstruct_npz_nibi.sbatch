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
: "${INPUT_NAME:=imagenet_val_50k_256_balanced.npz}"
: "${OUTPUT_NAME:=imagenet_val_50k_256_balanced_reconstructed_sdvae.npz}"

module load StdEnv/2023
module load python/3.11
module load gcc
module load opencv/4.14.0

source ~/projects/def-kdhkdh/jiaqi217/envs/fdloss/bin/activate

export project=~/projects/def-kdhkdh/jiaqi217
export REPR_REPO=$project/Repr_Research_Project
export EVAL_REFS=$project/data/eval_refs
export HF_HOME=$SCRATCH/huggingface
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$REPR_REPO"
mkdir -p logs

python - <<'PY'
import torch

print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY

INPUT_NPZ="$EVAL_REFS/$INPUT_NAME"
OUTPUT_NPZ="$EVAL_REFS/$OUTPUT_NAME"

cmd=(
  python reconstruction/reconstruct_npz.py
  --input-npz "$INPUT_NPZ"
  --output-npz "$OUTPUT_NPZ"
  --batch-size "$BATCH_SIZE"
)
if [[ -n "$MAX_IMAGES" ]]; then
  cmd+=(--max-images "$MAX_IMAGES")
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"

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
