# MAE slow-critic B/C line

This is a new two-arm experiment; existing Inception B/C and their checkpoints and evaluations remain unchanged. Both arms reconstruct ImageNet at 256 px with the same pretrained SD-VAE, E+D optimizer, data order, seed, batch size 16, 10,000 G updates, dynamic MAE initialization, D learning rate 2e-5, and D-then-G ordering. Both first update the dynamic feature extractor after 3,125 G updates (50,000 images), then once per 3,125 updates. The feature extractor is fixed between those updates. This schedule is a research variant, **not** the published AdvFD MAE schedule.

| Arm | Frozen static FD | Dynamic MAE | Dynamic statistics |
| --- | --- | --- | --- |
| B | Official SIM (Inception, SigLIP, MAE) | QKV rank-16 LoRA | Official real/fake EMA |
| C | None | All MAE parameters | Real 50k pool re-encoded under current MAE; fake statistics from the current 16-image reconstruction batch, without EMA |

C's real pool is refreshed after each MAE update and stays valid while MAE is fixed. E+D changes every batch, so a cached fake 50k pool would become stale immediately; this variant deliberately uses only the current batch of fake images. Its batch covariance has rank at most 15 and may be noisy. It must **not** be described as the existing C's exact current 50k fake-pool FD or full-pool gradient. The held-out scientific evaluation still uses 50k images and frozen independent extractors. The two training losses are not directly comparable in scale.

Both arms share the same E+D update count and 16-image G batch, unlike the old Inception B/C comparison. The feature-update cadence is matched, but B's real/fake EMA and C's 50k-real/16-fake current estimator necessarily use different statistics. The new C is an explicitly labelled approximation; positive validation results, not the training FD, are required to support the hypothesis.

For CUDA training with strict PyTorch determinism, timm's 256-to-224 bicubic
antialias resize has no deterministic backward kernel. The forward resize is
unchanged; only its input-gradient computation temporarily opts out of strict
determinism, then restores the prior setting. This exception is recorded in
provenance. It does not make the complete run bitwise reproducible.

Use `jobs/mae_slow_nibi.sbatch` with `FD_GROUP=B|C` and `FD_SCALE=preflight|full`. The preflight reduces reference pools to 128 and brings the first D update forward to step 1; it verifies engineering only and is not a 50k research result. Run the arms independently and sequentially on one GPU, with code synchronized through GitHub. Do not update the existing checkout while pinned evaluation jobs depend on its earlier commit.
