#!/usr/bin/env bash
# Four separate 50k validation checkpoints. Submit with afterany dependencies.
#SBATCH --job-name=recon-fd-50k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
set -euo pipefail

: "${SLURM_JOB_ID:?Submit through sbatch}"
: "${FD_PROJECT:?Absolute clean checkout path}"
: "${FD_PYTHON:?Cluster Python executable}"
: "${FD_RUN_ROOT:?Existing experiment root}"
: "${FD_EXPECTED_COMMIT:?Reviewed Git commit}"
: "${FD_GROUP:?B or C}"
: "${FD_STEP:?Checkpoint step}"
: "${IMAGENET_ROOT:?ImageNet root}"
: "${TOKENIZER_CHECKPOINT:?AutoencoderKL checkpoint}"
: "${INCEPTION_WEIGHTS:?Official Inception weights}"

case "$FD_GROUP:$FD_STEP" in
  B:3000|B:10000) config=configs/advfd_nibi_large.yaml ;;
  C:0|C:4) config=configs/ours_current_both_nibi_large.yaml ;;
  *) echo "Unsupported checkpoint: $FD_GROUP:$FD_STEP" >&2; exit 2 ;;
esac
for path in "$FD_PROJECT" "$FD_PYTHON" "$FD_RUN_ROOT" "$IMAGENET_ROOT" "$TOKENIZER_CHECKPOINT" "$INCEPTION_WEIGHTS"; do
  case "$path" in /*) ;; *) echo "Expected absolute path: $path" >&2; exit 2 ;; esac
done
test -x "$FD_PYTHON"
test -d "$IMAGENET_ROOT/val"
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

printf -v checkpoint '%s/full/%s/initial/checkpoints/step_%07d.pt' "$FD_RUN_ROOT" "$FD_GROUP" "$FD_STEP"
output="$FD_RUN_ROOT/evaluation/fixed50k_inception_dinov2_v1/$FD_GROUP/step_$FD_STEP"
test -f "$checkpoint"
if [[ -e "$output" ]]; then
  echo "Refusing existing evaluation output: $output" >&2
  exit 2
fi

representations='[{"name":"inception","kind":"inception","weights":"'"$INCEPTION_WEIGHTS"'","pool":"cls","weight":1.0},{"name":"dinov2","kind":"timm","model_name":"vit_large_patch14_dinov2.lvd142m","target_size":256,"pool":"cls","weight":1.0}]'
echo "start_utc=$(date -u +%FT%TZ) job=$SLURM_JOB_ID group=$FD_GROUP step=$FD_STEP commit=$FD_EXPECTED_COMMIT host=$(hostname)"
echo "checkpoint=$checkpoint output=$output config=$config"
echo "protocol=ImageNet-val-50000 frozen_inception+dinov2 psnr+ssim+residual_energy"
echo "lpips=omitted (cluster LPIPS/AlexNet pretrained weights not installed)"
sha256sum "$checkpoint"
nvidia-smi

# Fail before exporting 50k PNGs if the cached independent evaluator is unavailable.
"$FD_PYTHON" -c 'import skimage,sys; from recon_fd.representations import build_representation
r=build_representation({"name":"dinov2","kind":"timm","model_name":"vit_large_patch14_dinov2.lvd142m","target_size":256,"pool":"cls"})
print("independent_evaluator_ready",r.dimension,r.fingerprint,flush=True)'

"$FD_PYTHON" evaluate.py --config "$config" \
  --set evaluation.num_samples=50000 \
  --set evaluation.engineering_only=false \
  --set "evaluation.representations=$representations" \
  --set 'evaluation.paired_metrics=["psnr","ssim"]' \
  --set "static.reference_cache=$FD_RUN_ROOT/cache/static" \
  --checkpoint "$checkpoint" --output "$output"

"$FD_PYTHON" -c 'import csv,json,sys; from pathlib import Path
p=Path(sys.argv[1]); step=int(sys.argv[2]); m=json.loads((p/"metrics.json").read_text()); e=json.loads((p/"reconstructions/complete.json").read_text()); n=sum(1 for _ in csv.DictReader((p/"per_image.csv").open()))
assert m["num_samples"]==e["count"]==n==50000, (m["num_samples"],e["count"],n)
assert m["checkpoint_step"]==step and set(m["fd"])=={"inception","dinov2"}
assert set(m["paired"])=={"psnr","ssim"}
assert m["dataset"]==e["identity"]["dataset"]
print("verified_50k",json.dumps({"step":step,"count":n,"fd":m["fd"],"paired":m["paired"],"model_sha256":m["checkpoint_model_sha256"]},sort_keys=True),flush=True)' "$output" "$FD_STEP"
echo "end_utc=$(date -u +%FT%TZ) job=$SLURM_JOB_ID status=verified"
