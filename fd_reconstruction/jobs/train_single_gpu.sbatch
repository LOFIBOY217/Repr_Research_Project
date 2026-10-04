#!/usr/bin/env bash
# Supply your account/partition explicitly at submission time.
#SBATCH --job-name=recon-fd-only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=recon-fd-%j.out
set -euo pipefail
: "${FD_PROJECT:?Set absolute path to fd_reconstruction}"
: "${FD_PYTHON:?Set the cluster environment Python executable}"
: "${IMAGENET_ROOT:?Set ImageNet root containing train and val}"
: "${TOKENIZER_CHECKPOINT:?Set pretrained AutoencoderKL checkpoint directory}"
cd "$FD_PROJECT"
exec "$FD_PYTHON" train.py --config "${FD_CONFIG:-configs/fd_only_inception.yaml}" "$@"
