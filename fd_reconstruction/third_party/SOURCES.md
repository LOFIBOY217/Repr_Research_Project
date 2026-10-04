# Source snapshots

These directories are read-only reference snapshots, not runtime dependencies. No nested Git repositories are required.

| Directory | Source | Pinned identity |
| --- | --- | --- |
| `FD-Loss` | https://github.com/Jiawei-Yang/FD-Loss | `5c03b8112fec8b9432631e4ce053c0d918cc24bc` |
| `AdvFD` | https://github.com/GasaiYU/AdvFD | `4e4cfed944e4fc38a75fae3ea7701ae9e5587060` |
| `Grounded-Frechet-Loss-main` (local only, excluded from Git) | User-provided ZIP | SHA256 `247d5486ff1d497431331af4cb0c8e907f662dec70efca6a981efcefde2583b5` |

Official repositories were exported with `git archive`; the ZIP was extracted without modifying its files. The old pilot's FD-Loss commit spelling and AdvFD placeholder are not used here.

FD-Loss and AdvFD retain their upstream LICENSE files. No top-level license was present in the supplied Grounded ZIP; local availability is not evidence of permission to publicly redistribute it. Confirm the author's terms before publishing that snapshot.

The runtime Inception architecture and FD/EMA/queue conventions derive from FD-Loss; A now calls unchanged copies of the pinned losses.py and queue.py under recon_fd/vendor/fd_loss. The KL reconstruction adapter follows Grounded's posterior-mean encode/decode and clamp convention without a runtime dependency on the local archive. B implements the AdvFD reconstruction adaptation documented in ADVFD_BASELINE.md. The candidate C is not implemented.
