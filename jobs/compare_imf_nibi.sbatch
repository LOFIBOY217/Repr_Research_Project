#!/bin/bash
#SBATCH --job-name=gen-imf-cmp
#SBATCH --account=def-kdhkdh
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

: "${IMF_CKPT_PATH:?Set IMF_CKPT_PATH to the iMF checkpoint .pth}"
: "${IMF_XL_CKPT_PATH:?Set IMF_XL_CKPT_PATH to the iMF-XL checkpoint .pth}"
: "${NUM_IMAGES:=8}"
: "${BATCH_SIZE:=2}"
: "${OUTPUT_SUBDIR:=generation/imf_vs_imf-xl_smoke}"

module load StdEnv/2023
module load python/3.11
module load gcc
module load opencv/4.14.0

source ~/projects/def-kdhkdh/jiaqi217/envs/fdloss/bin/activate

export project=~/projects/def-kdhkdh/jiaqi217
export REPR_REPO=$project/Repr_Research_Project
export REPR_SCRATCH=$SCRATCH/Repr_Research_Project
export HF_HOME=$SCRATCH/huggingface
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$REPR_REPO"
mkdir -p "$REPR_SCRATCH/$OUTPUT_SUBDIR"

python - <<'PY'
import torch

print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY

python generation/generate_images.py \
  --preset imf \
  --checkpoint "$IMF_CKPT_PATH" \
  --compare-preset imf-xl \
  --compare-checkpoint "$IMF_XL_CKPT_PATH" \
  --output-dir "$REPR_SCRATCH/$OUTPUT_SUBDIR" \
  --num-images "$NUM_IMAGES" \
  --batch-size "$BATCH_SIZE" \
  --num-sampling-steps 1 \
  --save-grid

echo "Output directory:"
ls -lh "$REPR_SCRATCH/$OUTPUT_SUBDIR"
