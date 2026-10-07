#!/usr/bin/env bash
# CPU-only qualitative preview; no new training or FD claims.
#SBATCH --job-name=recon-BC-visual
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail

: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${FD_PROJECT:?Pinned evaluator source checkout}"
: "${FD_SCRIPT_PROJECT:?Checkout containing this versioned script}"
: "${FD_PYTHON:?Cluster Python executable}"
: "${FD_RUN_ROOT:?Existing experiment root}"
: "${FD_EXPECTED_COMMIT:?Pinned evaluator source commit}"
: "${FD_SCRIPT_COMMIT:?Versioned script commit}"
: "${IMAGENET_ROOT:?ImageNet root}"
test "$(git -C "${FD_PROJECT%/fd_reconstruction}" rev-parse HEAD)" = "$FD_EXPECTED_COMMIT"
test "$(git -C "${FD_SCRIPT_PROJECT%/fd_reconstruction}" rev-parse HEAD)" = "$FD_SCRIPT_COMMIT"

module load StdEnv/2023
module load python/3.11
module load gcc
module load opencv/4.14.0
export PYTHONPATH="$FD_PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1

output="$FD_RUN_ROOT/visual/BC/curated36_seed217"
b="$FD_RUN_ROOT/full/B/initial/checkpoints/step_0010000.pt"
c="$FD_RUN_ROOT/full/C/initial/checkpoints/step_0000004.pt"
test -f "$b"
test -f "$c"
test ! -e "$output"
echo "job=$SLURM_JOB_ID source_commit=$FD_EXPECTED_COMMIT script_commit=$FD_SCRIPT_COMMIT output=$output"
"$FD_PYTHON" "$FD_SCRIPT_PROJECT/scripts/curated_bc_visual.py" \
  --val-root "$IMAGENET_ROOT/val" --checkpoint-b "$b" --checkpoint-c "$c" \
  --output "$output" --per-class 6 --seed 217
"$FD_PYTHON" -c 'import json,sys; from pathlib import Path
p=Path(sys.argv[1]); m=json.loads((p/"manifest.json").read_text())
assert m["count"]==36 and len(m["classes"])==6
assert all((p/item["synset"]/arm/("{:02d}.png".format(item["index"]))).is_file()
           for item in m["samples"] for arm in ("original","B","C"))
assert all((p/(item["synset"]+".png")).is_file() for item in m["classes"])
print("verified_curated_BC_36_images_6_sheets",flush=True)' "$output"
