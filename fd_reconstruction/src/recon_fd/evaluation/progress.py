"""Small, timestamped progress messages for long single-GPU evaluations."""
from datetime import datetime, timezone


def log_progress(stage, done, total):
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{stamp}] eval {stage}: {done}/{total}", flush=True)


def should_log(done, previous, total, interval=5000):
    return done == total or done // interval > previous // interval
