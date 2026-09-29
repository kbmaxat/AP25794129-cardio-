"""Aggregate the resolution-robustness screening: patient-level averages (ED/ES) per
variant, paired contrasts (clean vs degraded, plain vs sharpened), and patient-level
percentile bootstrap 95% CIs. Descriptive only: no hypothesis test, no confirmatory
threshold (see resolution_robustness_screening_protocol_v01.md).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PACKAGE = ROOT / "ap25794129-cardio-preprocessing"
sys.path.insert(0, str(PACKAGE))

import numpy as np

from cardiac_image_system.research_temporal.data import sha256

METRICS = (
    "dice",
    "relative_area_error",
    "mean_surface_distance_px",
    "hausdorff_distance_px",
    "spurious_components",
)
VARIANTS = ("clean", "2x_plain", "2x_sharpened", "4x_plain", "4x_sharpened")
CONTRASTS = (
    ("2x_plain", "clean"),
    ("4x_plain", "clean"),
    ("2x_sharpened", "2x_plain"),
    ("4x_sharpened", "4x_plain"),
)


def patient_level(rows: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["patient"], row["variant"]), []).append(row)
    patients = sorted({patient for patient, _ in grouped})
    result: dict[str, dict[str, dict[str, float]]] = {variant: {} for variant in VARIANTS}
    for variant in VARIANTS:
        for patient in patients:
            phases = grouped.get((patient, variant))
            if not phases or len(phases) != 2:
                raise ValueError(f"Expected exactly 2 phases for {patient}/{variant}")
            result[variant][patient] = {
                metric: sum(float(row[metric]) for row in phases) / len(phases)
                for metric in METRICS
            }
    return result


def percentile_bootstrap(values: np.ndarray, seed: int, replicates: int = 10000) -> list[float]:
    if values.ndim != 1 or values.shape[0] < 2:
        raise ValueError("Bootstrap requires at least two patients")
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        selected = rng.integers(0, n, n)
        estimates[index] = values[selected].mean()
    return [float(v) for v in np.quantile(estimates, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path, required=True)
    args = parser.parse_args()

    private_output = args.private_output.resolve()
    aggregate_output = args.aggregate_output.resolve()
    metadata = json.loads((private_output / "metadata.json").read_text(encoding="utf-8"))
    with (private_output / "resolution_robustness_phase_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("No phase-level rows")

    per_variant = patient_level(rows)
    patients = sorted(next(iter(per_variant.values())))
    if len(patients) != metadata["n_train_patients"]:
        raise ValueError("Patient count does not match run metadata")

    result = {
        "experiment_id": metadata["experiment_id"],
        "stage": metadata["stage"],
        "n_train_patients": len(patients),
        "test_access": metadata["test_access"],
        "p_values_computed": False,
        "variant_means": {},
        "contrasts": {},
        "checkpoint_run": metadata["checkpoint_run"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "protocol_sha256": metadata["protocol_sha256"],
        "patient_level_csv_sha256": sha256(
            private_output / "resolution_robustness_phase_metrics.csv"
        ),
    }
    for variant in VARIANTS:
        result["variant_means"][variant] = {
            metric: float(np.mean([per_variant[variant][p][metric] for p in patients]))
            for metric in METRICS
        }
    for contrast_index, (treated, reference) in enumerate(CONTRASTS):
        name = f"{treated}_minus_{reference}"
        result["contrasts"][name] = {}
        for metric_index, metric in enumerate(METRICS):
            delta = np.array([
                per_variant[treated][p][metric] - per_variant[reference][p][metric]
                for p in patients
            ])
            result["contrasts"][name][metric] = {
                "mean_delta": float(delta.mean()),
                "patient_bootstrap_95_ci": percentile_bootstrap(
                    delta, seed=303000 + contrast_index * 10 + metric_index
                ),
            }

    aggregate_output.mkdir(parents=True, exist_ok=False)
    summary_path = aggregate_output / "resolution_robustness_aggregate.json"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    public_metadata = {
        "experiment_id": metadata["experiment_id"],
        "stage": metadata["stage"],
        "protocol_sha256": metadata["protocol_sha256"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "private_metadata_sha256": sha256(private_output / "metadata.json"),
        "private_phase_metrics_sha256": result["patient_level_csv_sha256"],
        "test_access": metadata["test_access"],
        "patient_level_rows_included": False,
    }
    (aggregate_output / "provenance.json").write_text(
        json.dumps(public_metadata, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    hashes = {
        path.name: sha256(path)
        for path in sorted(aggregate_output.iterdir())
        if path.is_file()
    }
    (aggregate_output / "files_sha256.json").write_text(
        json.dumps(hashes, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=True))
    print(f"OUTPUT={aggregate_output}")


if __name__ == "__main__":
    main()
