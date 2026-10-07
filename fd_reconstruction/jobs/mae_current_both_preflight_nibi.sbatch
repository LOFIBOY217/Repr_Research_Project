#!/usr/bin/env bash
# Exact paired-pool MAE-C engineering check only; one H100, no formal 50k claim.
#SBATCH --job-name=recon-mae-C-exact-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail

: "${SLURM_JOB_ID:?Submit through sbatch, not on the login node}"
: "${FD_PROJECT:?Absolute clean GitHub checkout}"
: "${FD_PYTHON:?Absolute cluster Python executable}"
: "${FD_RUN_ROOT:?Fresh scratch experiment root}"
: "${FD_EXPECTED_COMMIT:?Reviewed Git commit}"
: "${IMAGENET_ROOT:?ImageNet train/val root}"
: "${TOKENIZER_CHECKPOINT:?AutoencoderKL checkpoint}"
for path in "$FD_PROJECT" "$FD_PYTHON" "$FD_RUN_ROOT" "$IMAGENET_ROOT" "$TOKENIZER_CHECKPOINT"; do
  case "$path" in /*) ;; *) echo "Expected absolute path: $path" >&2; exit 2 ;; esac
done
test -x "$FD_PYTHON"
test -d "$IMAGENET_ROOT/train"
test -d "$IMAGENET_ROOT/val"
test -f "$TOKENIZER_CHECKPOINT/config.json"

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
run_dir="$FD_RUN_ROOT/preflight/C"
if [[ -e "$run_dir" ]]; then
  echo "Refusing existing run directory: $run_dir" >&2
  exit 2
fi
mkdir -p "$(dirname "$run_dir")"
echo "start_utc=$(date -u +%FT%TZ) job=$SLURM_JOB_ID commit=$FD_EXPECTED_COMMIT output=$run_dir"
echo "protocol=exact_current_real_and_fake paired_pool_128 microbatch_16 G_2 D_1 engineering_only"
nvidia-smi
"$FD_PYTHON" train.py --config configs/mae_current_both_c.yaml \
  --set "train.output=$run_dir" \
  --set evaluation.engineering_only=true \
  --set adaptive.real_stats.samples=128 \
  --set adaptive.initialization_samples=128 \
  --set train.steps=2 \
  --set train.save_every=1 \
  --set train.log_every=1
"$FD_PYTHON" - "$run_dir" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
rows = [json.loads(line) for line in (run / "train.jsonl").read_text().splitlines()]
assert len(rows) == 2 and [row["step"] for row in rows] == [1, 2]
assert [row["critic_updates"] for row in rows] == [1, 1]
assert [row["generator_updates"] for row in rows] == [1, 2]
for row in rows:
    assert row["real_reference_pool_samples"] == row["fake_reference_pool_samples"] == 128
    assert row["fd_batch_size"] == 128 and row["feature_microbatch_size"] == 16
    assert row["real_ema_updates"] == row["fake_ema_updates"] == 0
    assert row["real_feature_version"] == row["fake_feature_psi_version"] == row["loss_psi_version"]
    assert row["fake_feature_generator_version"] == row["loss_generator_version"]
    assert row["gradient_estimator"] == "full_pool_two_pass_chain_rule"
assert (run / "checkpoints/step_0000002.pt").is_file()
print("verified_exact_pool_mae_preflight", json.dumps({"steps": 2, "critic_updates": 1,
      "seconds": [row["step_seconds"] for row in rows]}), flush=True)
PY
echo "end_utc=$(date -u +%FT%TZ) job=$SLURM_JOB_ID status=verified"
