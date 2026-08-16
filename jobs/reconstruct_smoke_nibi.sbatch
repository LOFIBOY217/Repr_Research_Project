#!/bin/bash
#SBATCH --job-name=recon-smoke
#SBATCH --account=def-kdhkdh
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load StdEnv/2023
module load python/3.11
module load gcc
module load opencv/4.14.0

source ~/projects/def-kdhkdh/jiaqi217/envs/fdloss/bin/activate

export project=~/projects/def-kdhkdh/jiaqi217
export REPR_REPO=$project/Repr_Research_Project
export REPR_SCRATCH=$SCRATCH/Repr_Research_Project
export DATA_ROOT=/datashare/imagenet/ILSVRC2012
export HF_HOME=$SCRATCH/huggingface
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$REPR_REPO"
mkdir -p "$REPR_SCRATCH/reconstruction/smoke_n01440764"

python - <<'PY'
import torch

print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY

python reconstruction/reconstruct_images.py \
  --input-dir "$DATA_ROOT/val/n01440764" \
  --output-dir "$REPR_SCRATCH/reconstruction/smoke_n01440764" \
  --img-size 256 \
  --batch-size 2 \
  --max-images 8 \
  --num-workers 4 \
  --save-inputs \
  --save-comparisons

echo "Output directory:"
ls -lh "$REPR_SCRATCH/reconstruction/smoke_n01440764"
