# B/C fixed 50k validation, first pass

Evaluate B steps 3000 and 10000, and C steps 0 and 4, on the same first 50,000 ImageNet validation images. These are four independent checkpoint evaluations, serialized only to use one GPU at a time. They do not resume or alter training.

Each run exports 50,000 uint8 PNG reconstructions, then measures FD with frozen official Inception and an independent frozen DINOv2 ViT-L. The same cached real-image reference statistics, preprocessing, sample IDs, and metric definitions are used for all four runs. Paired PSNR, SSIM, pixel MSE, and residual Laplacian energy are recorded per image; the first two are also averaged in `metrics.json`. LPIPS is explicitly excluded because the cluster LPIPS/AlexNet pretrained weights are not installed. A lower FD in only one feature space is not evidence that visual quality improved.

The job requests one H100, eight CPUs, 64 GiB host memory, and a 12-hour walltime limit per checkpoint. This is a conservative allocation, not a runtime prediction. The observed 12-image PNG preview averages about 119 KB/image, implying approximately 5.5 GiB of exported PNGs per checkpoint and 22 GiB for all four, before CSV and reference caches. Logs report the job, Git commit, checkpoint hash, chosen metrics, GPU, stage progress every 5,000 images, final measurements, and a post-run 50k/sample-count validation.

Outputs: `<FD_RUN_ROOT>/evaluation/fixed50k_inception_dinov2_v1/<B|C>/step_<n>/`. The PNG `complete.json` binds sample order and model hash; `metrics.json` binds dataset and feature identities, implementation hash, step, and metric values. `per_image.csv` enables later artifact screening. Training-log B static FD, B dynamic FD, and C dynamic FD are not interchangeable with these fixed validation FDs; C has only four full-pool updates and cannot be budget-matched to B's 10,000 minibatch updates.

## Matched A extension

Evaluate A checkpoints 3000 and 10000 under this same 50k validation protocol; outputs live under the sibling `A/step_<n>/` paths. The evaluator source checkout remains pinned to commit `9e418e30961fbe0ef52aec43aa8d445d8aa2e55b`, exactly as for B. A later version of the SLURM wrapper adds only `A` checkpoint/config routing; it does not change the evaluator implementation. A and B results can be compared only after verifying equal dataset, reference feature identities, sample counts, image protocol and evaluator implementation hashes in their `metrics.json` files.
