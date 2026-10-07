#!/usr/bin/env bash
# Read-only checkpoint audit; submit with one H100 after the corresponding preflight.
#SBATCH --job-name=recon-mae-psi-audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail

: "${SLURM_JOB_ID:?Submit through sbatch}"
: "${FD_PROJECT:?Absolute audit-code checkout}"
: "${FD_PYTHON:?Absolute cluster Python executable}"
: "${FD_SOURCE_RUN:?Completed preflight run directory}"
: "${FD_AUDIT_OUTPUT:?Fresh audit JSON path}"
: "${FD_EXPECTED_COMMIT:?Reviewed audit-code commit}"
for path in "$FD_PROJECT" "$FD_PYTHON" "$FD_SOURCE_RUN" "$FD_AUDIT_OUTPUT"; do
  case "$path" in /*) ;; *) echo "Expected absolute path: $path" >&2; exit 2 ;; esac
done
test -x "$FD_PYTHON"
test -f "$FD_SOURCE_RUN/checkpoints/step_0000000.pt"
test -f "$FD_SOURCE_RUN/checkpoints/step_0000001.pt"
test -f "$FD_SOURCE_RUN/checkpoints/step_0000002.pt"
if [[ -e "$FD_AUDIT_OUTPUT" ]]; then
  echo "Refusing existing audit result: $FD_AUDIT_OUTPUT" >&2
  exit 2
fi

module load StdEnv/2023
module load python/3.11
module load gcc
module load opencv/4.14.0
export PYTHONPATH="$FD_PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
cd "$FD_PROJECT"
test "$(git rev-parse HEAD)" = "$FD_EXPECTED_COMMIT"
git diff --quiet
git diff --cached --quiet
mkdir -p "$(dirname "$FD_AUDIT_OUTPUT")"
echo "start_utc=$(date -u +%FT%TZ) job=$SLURM_JOB_ID commit=$FD_EXPECTED_COMMIT source=$FD_SOURCE_RUN output=$FD_AUDIT_OUTPUT"
nvidia-smi
"$FD_PYTHON" scripts/audit_dynamic_extractor.py \
  --run "$FD_SOURCE_RUN" --output "$FD_AUDIT_OUTPUT" --device cuda --probe-count 8
echo "end_utc=$(date -u +%FT%TZ) job=$SLURM_JOB_ID status=verified"
