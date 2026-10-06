"""Export allowlisted aggregate metrics without private paths or video identifiers."""

import argparse
import hashlib
import json
from pathlib import Path


def export(source: Path, destination: Path) -> None:
    raw = source.read_bytes()
    summary = json.loads(raw)
    result = {
        "schema_version": 1,
        "stage": "post_hoc_diagnostic",
        "dataset": "EchoNet-Dynamic",
        "partition": "previously_examined_VAL",
        "adapter_version": "camus-echonet-axis-transpose-v1",
        "source_summary_sha256": hashlib.sha256(raw).hexdigest(),
        "training_performed": False,
        "test_access": False,
        "n_videos": summary["n_videos"],
        "n_seeds": summary["n_seeds"],
        "bootstrap_replicates": summary["bootstrap_replicates"],
        "candidate_configs": {},
    }
    metrics = {"dice", "relative_area_error", "mean_surface_distance_px", "hausdorff_distance_px"}
    allowed_means = {
        "clean_baseline_clean", "clean_baseline_degraded",
        "augmentation_clean", "augmentation_degraded",
    }
    for candidate in ("p025_2x", "p075_2x", "p050_4x"):
        config = summary["candidate_configs"][candidate]
        result["candidate_configs"][candidate] = {
            "degraded_condition": "4x_plain" if candidate == "p050_4x" else "2x_plain",
            "means": {
                name: {metric: float(config["means"][name][metric]) for metric in sorted(metrics)}
                for name in sorted(allowed_means)
            },
        }
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(serialized)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.output)
