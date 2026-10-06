#!/usr/bin/env bash
# New MAE B/C line only; no implicit combined run.
#SBATCH --job-name=recon-mae-slow
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
set -euo pipefail

: "${SLURM_JOB_ID:?Submit through sbatch}"
: "${FD_PROJECT:?Absolute clean checkout path}"
: "${FD_PYTHON:?Cluster Python executable}"
: "${FD_RUN_ROOT:?Fresh scratch experiment root}"
: "${FD_EXPECTED_COMMIT:?Reviewed Git commit}"
: "${FD_GROUP:?B or C}"
: "${FD_SCALE:?preflight or full}"
: "${IMAGENET_ROOT:?ImageNet root}"
: "${TOKENIZER_CHECKPOINT:?AutoencoderKL checkpoint}"
: "${INCEPTION_WEIGHTS:?Official Inception weights}"

case "$FD_GROUP" in
  B) config=configs/mae_slow_b.yaml ;;
  C) config=configs/mae_slow_c.yaml ;;
  *) echo "FD_GROUP must be B or C" >&2; exit 2 ;;
esac
case "$FD_SCALE" in preflight|full) ;; *) echo "FD_SCALE must be preflight or full" >&2; exit 2 ;; esac
for path in "$FD_PROJECT" "$FD_PYTHON" "$FD_RUN_ROOT" "$IMAGENET_ROOT" "$TOKENIZER_CHECKPOINT" "$INCEPTION_WEIGHTS"; do
  case "$path" in /*) ;; *) echo "Expected absolute path: $path" >&2; exit 2 ;; esac
done
test -x "$FD_PYTHON"
test -d "$IMAGENET_ROOT/train"
test -f "$TOKENIZER_CHECKPOINT/config.json"
test -f "$INCEPTION_WEIGHTS"

module load StdEnv/2023
module load python/3.11
module load gcc
module load opencv/4.14.0
export PYTHONPATH="$FD_PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

cd "$FD_PROJECT"
test "$(git rev-parse HEAD)" = "$FD_EXPECTED_COMMIT"
git diff --quiet
git diff --cached --quiet
run_dir="$FD_RUN_ROOT/$FD_SCALE/$FD_GROUP"
if [[ -e "$run_dir" ]]; then
  echo "Refusing existing run directory: $run_dir" >&2
  exit 2
fi
mkdir -p "$(dirname "$run_dir")" "$FD_RUN_ROOT/cache/static"
args=(--set "train.output=$run_dir" --set "static.reference_cache=$FD_RUN_ROOT/cache/static")
if [[ "$FD_SCALE" == preflight ]]; then
  args+=(--set evaluation.engineering_only=true
         --set static.reference_samples=128
         --set static.initialization_samples=128
         --set static.queue_size=128
         --set train.steps=2
         --set train.save_every=1
         --set train.log_every=1
         --set adaptive.first_critic_step=1
         --set adaptive.update_freq=2)
  if [[ "$FD_GROUP" == C ]]; then
    args+=(--set adaptive.real_stats.samples=128
           --set adaptive.initialization_samples=128)
  fi
fi
echo "start_utc=$(date -u +%FT%TZ) job=$SLURM_JOB_ID group=$FD_GROUP scale=$FD_SCALE commit=$FD_EXPECTED_COMMIT output=$run_dir"
if [[ "$FD_SCALE" == preflight ]]; then
  echo "protocol=engineering_only G_batch16 D_at_second_step reference_128"
else
  echo "protocol=G_batch16 D_every_50000_images first_D_after_50000"
fi
nvidia-smi
if [[ "$FD_SCALE" == preflight ]]; then
  "$FD_PYTHON" -c 'import torch
from torch.nn import functional as F
from recon_fd.representations import _resize_bicubic_antialias
from recon_fd.runtime import configure_determinism
configure_determinism("cuda")
x=torch.randn(2,3,256,256,device="cuda",requires_grad=True)
y=_resize_bicubic_antialias(x,(224,224))
expected=F.interpolate(x.detach(),(224,224),mode="bicubic",align_corners=False,antialias=True)
assert torch.equal(y,expected)
y.square().mean().backward()
assert torch.isfinite(x.grad).all() and torch.are_deterministic_algorithms_enabled()
print("verified_cuda_bicubic_aa_forward_and_scoped_backward",flush=True)'
fi
"$FD_PYTHON" train.py --config "$config" "${args[@]}"
"$FD_PYTHON" -c 'import json,sys; from pathlib import Path
p=Path(sys.argv[1]); n=int(sys.argv[2]); rows=[json.loads(s) for s in (p/"train.jsonl").open()]; assert len(rows)==n and rows[-1]["step"]==n
assert rows[-1]["fd_batch_size"]==16 and all(row["samples_seen"]==row["step"]*16 for row in rows)
if n==2: assert rows[-1]["critic_updates"]==1
assert (p/f"checkpoints/step_{n:07d}.pt").is_file()
print("verified_training",json.dumps({"steps":n,"critic_updates":rows[-1]["critic_updates"],"samples_seen":rows[-1]["samples_seen"]}),flush=True)' "$run_dir" "$([[ "$FD_SCALE" == preflight ]] && echo 2 || echo 10000)"
echo "end_utc=$(date -u +%FT%TZ) job=$SLURM_JOB_ID status=verified"
