#!/bin/bash
#SBATCH --job-name=eval-npz
#SBATCH --account=def-kdhkdh
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

: "${REF_NAME:=imagenet_val_50k_256_balanced.npz}"
: "${SAMPLE_NAME:=imagenet_val_50k_256_balanced_reconstructed_sdvae.npz}"

module load StdEnv/2023
module load python/3.11
module load gcc
module load opencv/4.14.0

source ~/projects/def-kdhkdh/jiaqi217/envs/fdloss/bin/activate

export project=~/projects/def-kdhkdh/jiaqi217
export REPR_REPO=$project/Repr_Research_Project
export EVAL_REFS=$project/data/eval_refs
export EVAL_CACHE=$EVAL_REFS/evaluator_cache
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TF_CPP_MIN_LOG_LEVEL=2

cd "$REPR_REPO"
mkdir -p logs "$EVAL_CACHE"

python - <<'PY'
import importlib
import torch

print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

for package in ("numpy", "scipy", "tensorflow"):
    module = importlib.import_module(package)
    print(f"{package}: {getattr(module, '__version__', 'unknown')}")
PY

REF_NPZ="$EVAL_REFS/$REF_NAME"
SAMPLE_NPZ="$EVAL_REFS/$SAMPLE_NAME"

if [[ ! -f "$REF_NPZ" ]]; then
  echo "Missing reference npz: $REF_NPZ" >&2
  exit 1
fi

if [[ ! -f "$SAMPLE_NPZ" ]]; then
  echo "Missing sample npz: $SAMPLE_NPZ" >&2
  exit 1
fi

echo "reference: $REF_NPZ"
echo "sample:    $SAMPLE_NPZ"
echo "cache:     $EVAL_CACHE"

# evaluator.py stores the Inception graph in the current working directory.
cd "$EVAL_CACHE"

python "$REPR_REPO/evaluator/evaluator.py" "$REF_NPZ" "$SAMPLE_NPZ"
