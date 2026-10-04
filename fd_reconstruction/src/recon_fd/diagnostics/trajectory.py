import json
from pathlib import Path


def compare_evaluations(paths, target, heldout, tolerance=0.01):
    """Compare fixed-evaluator FD trajectories, not changing online objectives.

    tolerance is a descriptive relative-change screen, NOT a calibrated
    significance threshold or evidence of perceptual degradation.
    """
    records = [json.loads(Path(path).read_text()) for path in paths]
    records.sort(key=lambda record: record["checkpoint_step"])
    if len(records) < 2 or not heldout or target in heldout or tolerance < 0:
        raise ValueError("Need >=2 checkpoints and distinct target/held-out representations")
    if len({r["checkpoint_step"] for r in records}) != len(records):
        raise ValueError("Duplicate checkpoint steps; compare one run at a time")
    first = records[0]
    for record in records:
        for key in ("dataset", "num_samples", "image_protocol", "representations", "engineering_only", "implementation_sha256"):
            if record[key] != first[key]:
                raise ValueError(f"Incomparable evaluation protocols: {key}")
        if any(name not in record["fd"] for name in [target, *heldout]):
            raise ValueError("Missing required evaluation representation")
    comparisons = []
    for before, after in zip(records, records[1:]):
        changes = {name: (after["fd"][name] - before["fd"][name]) / max(abs(before["fd"][name]), 1e-12)
                   for name in [target, *heldout]}
        comparisons.append({"from_step": before["checkpoint_step"], "to_step": after["checkpoint_step"],
                            "relative_fd_changes": changes,
                            "cross_representation_divergence_candidate": changes[target] < -tolerance
                                and any(changes[name] > tolerance for name in heldout),
                            "paired_before": before["paired"], "paired_after": after["paired"]})
    return {"engineering_only": first["engineering_only"], "relative_screening_tolerance": tolerance,
            "warning": "A screening candidate is not proof of visual artifacts; repeated evaluation and image audit are required.",
            "comparisons": comparisons}
