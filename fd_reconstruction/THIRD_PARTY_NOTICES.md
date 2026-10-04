# Third party code notices

FD-Loss is copyright 2026 FD-Loss Authors and licensed under MIT. Its full license is preserved at `third_party/FD-Loss/LICENSE` and packaged as `recon_fd/FD_LOSS_LICENSE.txt`.

`representations/inception.py` contains the Inception portion of upstream `utils/perception_util.py`, with unrelated dependencies removed. `objectives/frechet.py`, `objectives/statistics.py`, and `representations/__init__.py` adapt the upstream FD, moments, and feature-extraction conventions.

The A training baseline now directly calls `vendor/fd_loss/losses.py` and `vendor/fd_loss/queue.py`. These two files are byte-identical copies of `frechet_distance/losses.py` and `frechet_distance/queue.py` from FD-Loss commit `5c03b8112fec8b9432631e4ce053c0d918cc24bc`, including upstream dtypes, derivatives and numerical edge behavior. The MIT license above covers these copies. `objectives/official_fd.py` adapts initialization, reconstruction inputs and versioned commits without reimplementing the numerical core. Normalized loss assembly follows upstream `main_fd.py`.

The older adapted `objectives/frechet.py` and `statistics.py` remain used by B and/or independent evaluation, not by A's training FD core. Their changes include high-precision FD arithmetic, explicit failures instead of fallback losses and a finite chosen derivative at the singular square-root boundary. They are not byte-identical to upstream. The common trainer retains failure checks, provenance checks, explicit post-optimizer statistics commits and reconstruction-only I/O; these are not claimed to be an unchanged upstream training script.

Grounded-Frechet-Loss is a user-supplied research code archive. Its original files and authorship are preserved. The new KL adapter follows its reconstruction protocol. The archive contains no top-level license grant; do not infer permission to publish the archived code.

AdvFD's original MIT license is retained at `third_party/AdvFD/LICENSE` (the same FD-Loss Authors license text is packaged in `recon_fd/FD_LOSS_LICENSE.txt`). The B baseline ports whitening from `frechet_distance/adversarial.py`, fused-QKV LoRA from `frechet_distance/repr_models.py`, and the main training loop's update/initialization conventions from commit `4e4cfed944e4fc38a75fae3ea7701ae9e5587060`. These live in `objectives/whitening.py`, `objectives/adaptive_fd.py`, `representations/lora.py` and `engine/adversarial.py`. Reconstruction input, explicit statistics commits, complete checkpoints and failure checks are adaptations; `ADVFD_BASELINE.md` records paper/script differences. The candidate method C is not implemented.

Pretrained model weights and datasets retain their own terms and are not distributed with this project.
