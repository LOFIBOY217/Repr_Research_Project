#!/bin/bash
#SBATCH --job-name=gen-imf-npz
#SBATCH --account=def-kdhkdh
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

: "${PRESET:?Set PRESET to imf-b or imf-xl}"
: "${CKPT_PATH:?Set CKPT_PATH to the checkpoint .pth}"
: "${NUM_IMAGES:=1000}"
: "${NUM_CLASSES:=1000}"
: "${BATCH_SIZE:=1}"
: "${CFG:=8.0}"
: "${SEED:=0}"
: "${OUTPUT_NAME:=imagenet_${NUM_IMAGES}_${PRESET}_cfg${CFG}.npz}"

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
mkdir -p logs "$EVAL_REFS"

OUTPUT_NPZ="$EVAL_REFS/$OUTPUT_NAME"

python - <<'PY'
import torch

print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY

python generation/generate_npz.py \
  --preset "$PRESET" \
  --checkpoint "$CKPT_PATH" \
  --output-npz "$OUTPUT_NPZ" \
  --num-images "$NUM_IMAGES" \
  --num-classes "$NUM_CLASSES" \
  --batch-size "$BATCH_SIZE" \
  --cfg "$CFG" \
  --seed "$SEED"

OUTPUT_NPZ="$OUTPUT_NPZ" python - <<'PY'
import os
import numpy as np

path = os.environ["OUTPUT_NPZ"]
data = np.load(path)
arr = data["arr_0"]
print("output:", path)
print("files:", data.files)
print("arr_0 shape:", arr.shape)
print("arr_0 dtype:", arr.dtype)
print("arr_0 min/max:", arr.min(), arr.max())
PY
