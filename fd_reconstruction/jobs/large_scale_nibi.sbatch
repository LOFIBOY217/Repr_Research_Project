#!/usr/bin/env bash
# One independent A, B, or C run per job. Pass --account and --gpus=h100:1.
# Use --time=01:00:00 for preflight; full runs may need the 7-day GPU tier.
#SBATCH --job-name=recon-large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=recon-large-%j.out
set -euo pipefail

: "${SLURM_JOB_ID:?Submit through sbatch, never run training on the login node}"
: "${FD_PROJECT:?Absolute path to the clean fd_reconstruction checkout}"
: "${FD_PYTHON:?Absolute path to the cluster Python executable}"
: "${FD_RUN_ROOT:?Absolute path to a new scratch experiment root}"
: "${FD_EXPECTED_COMMIT:?Reviewed Git commit}"
: "${FD_GROUP:?Choose A, B, or C}"
: "${FD_SCALE:?Choose preflight or full}"
: "${IMAGENET_ROOT:?ImageNet root containing train and val}"
: "${TOKENIZER_CHECKPOINT:?Local AutoencoderKL checkpoint}"
: "${INCEPTION_WEIGHTS:?Local official Inception checkpoint}"

case "$FD_GROUP" in
  A) config=configs/fd_only_nibi_large.yaml ;;
  B) config=configs/advfd_nibi_large.yaml ;;
  C) config=configs/ours_current_both_nibi_large.yaml ;;
  *) echo 'FD_GROUP must be A, B, or C' >&2; exit 2 ;;
esac
case "$FD_SCALE" in
  preflight|full) ;;
  *) echo 'FD_SCALE must be preflight or full' >&2; exit 2 ;;
esac
case "$FD_PROJECT:$FD_PYTHON:$FD_RUN_ROOT:$IMAGENET_ROOT:$TOKENIZER_CHECKPOINT:$INCEPTION_WEIGHTS" in
  *'${'*|*'..'*) echo 'Unresolved or parent-relative path' >&2; exit 2 ;;
esac
for path in "$FD_PROJECT" "$FD_RUN_ROOT" "$IMAGENET_ROOT" "$TOKENIZER_CHECKPOINT" "$INCEPTION_WEIGHTS" "$FD_PYTHON"; do
  case "$path" in /*) ;; *) echo "Expected absolute path: $path" >&2; exit 2 ;; esac
done
test -x "$FD_PYTHON"
test -d "$IMAGENET_ROOT/train"
test -d "$IMAGENET_ROOT/val"
test -f "$TOKENIZER_CHECKPOINT/config.json"
test -f "$TOKENIZER_CHECKPOINT/diffusion_pytorch_model.safetensors"
test -f "$INCEPTION_WEIGHTS"

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

cd "$FD_PROJECT"
test "$(git rev-parse HEAD)" = "$FD_EXPECTED_COMMIT"
git diff --quiet
git diff --cached --quiet

if [[ "$FD_SCALE" == preflight ]]; then
  test -z "${FD_RESUME_CHECKPOINT:-}"
  run_dir="$FD_RUN_ROOT/preflight/$FD_GROUP"
  args=(--set evaluation.engineering_only=true
        --set evaluation.num_samples=32
        --set static.reference_samples=128
        --set static.initialization_samples=128
        --set static.queue_size=128
        --set train.steps=2
        --set train.save_every=1
        --set train.log_every=1)
  if [[ "$FD_GROUP" == B ]]; then
    # Engineering-only activation: exercise D at full microbatch size.
    args+=(--set adaptive.start_step=0 --set adaptive.warmup_steps=0)
  elif [[ "$FD_GROUP" == C ]]; then
    args+=(--set adaptive.initialization_samples=128
           --set adaptive.real_stats.samples=128)
  fi
else
  attempt="${FD_ATTEMPT:-initial}"
  [[ "$attempt" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid FD_ATTEMPT' >&2; exit 2; }
  run_dir="$FD_RUN_ROOT/full/$FD_GROUP/$attempt"
  args=()
  if [[ -n "${FD_RESUME_CHECKPOINT:-}" ]]; then
    test -f "$FD_RESUME_CHECKPOINT"
    args+=(--resume "$FD_RESUME_CHECKPOINT")
  fi
fi
if [[ -e "$run_dir" ]]; then
  echo "Refusing an existing run directory: $run_dir" >&2
  exit 2
fi
mkdir -p "$FD_RUN_ROOT/cache/static" "$(dirname "$run_dir")"
echo "job=$SLURM_JOB_ID group=$FD_GROUP scale=$FD_SCALE commit=$FD_EXPECTED_COMMIT output=$run_dir"
nvidia-smi
exec "$FD_PYTHON" train.py --config "$config" \
  --set "train.output=$run_dir" \
  --set "static.reference_cache=$FD_RUN_ROOT/cache/static" \
  "${args[@]}"
