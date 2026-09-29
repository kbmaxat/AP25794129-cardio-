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
    "mean_abs_correction",
    "max_abs_correction",
)


def crossed_bootstrap(values: np.ndarray, seed: int, replicates: int = 50000) -> list[float]:
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("Crossed bootstrap requires at least two seeds and two patients")
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        selected_seeds = rng.integers(0, values.shape[0], values.shape[0])
        selected_patients = rng.integers(0, values.shape[1], values.shape[1])
        estimates[index] = values[np.ix_(selected_seeds, selected_patients)].mean()
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate paired image-correction penalty results.")
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path, required=True)
    args = parser.parse_args()

    private_output = args.private_output.resolve()
    aggregate_output = args.aggregate_output.resolve()
    metadata = json.loads((private_output / "metadata.json").read_text(encoding="utf-8"))
    with (private_output / "correction_penalty_patient_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("No patient-level paired rows")
    conditions = {"standard_penalty", "no_correction_penalty"}
    grouped = {}
    for row in rows:
        key = (int(row["seed"]), row["patient"], row["condition"])
        if key in grouped:
            raise ValueError(f"Duplicate patient-condition row: {key}")
        grouped[key] = row
    seeds = sorted({key[0] for key in grouped})
    patients = sorted({key[1] for key in grouped})
    if len(seeds) != metadata["n_seeds"] or len(patients) != metadata["n_dev_patients"]:
        raise ValueError("Row counts do not match run metadata")
    expected = {
        (seed, patient, condition)
        for seed in seeds
        for patient in patients
        for condition in conditions
    }
    if set(grouped) != expected:
        raise ValueError("Paired conditions are incomplete or unbalanced")

    values = {
        condition: {
            metric: np.empty((len(seeds), len(patients)), dtype=np.float64)
            for metric in METRICS
        }
        for condition in conditions
    }
    for seed_index, seed in enumerate(seeds):
        for patient_index, patient in enumerate(patients):
            for condition in conditions:
                row = grouped[(seed, patient, condition)]
                for metric in METRICS:
                    values[condition][metric][seed_index, patient_index] = float(row[metric])

    result = {
        "experiment_id": metadata["experiment_id"],
        "stage": metadata["stage"],
        "baseline_results_previously_inspected": metadata["baseline_results_previously_inspected"],
        "n_unique_dev_patients": len(patients),
        "n_seeds": len(seeds),
        "test_access": metadata["test_access"],
        "p_values_computed": metadata["p_values_computed"],
        "contrast": "standard_penalty_minus_no_correction_penalty",
        "minimum_useful_benefit_reference": {
            "dice_increase": 0.01,
            "mean_surface_distance_reduction_pixels": 0.5,
        },
        "guardrails_reference": {
            "mean_surface_distance_increase_pixels": 0.5,
            "relative_area_error_increase_absolute": 0.01,
        },
        "condition_means": {},
        "paired_differences": {},
        "source_runs": metadata["run_records"],
        "patient_level_csv_sha256": sha256(
            private_output / "correction_penalty_patient_metrics.csv"
        ),
    }
    for condition_index, condition in enumerate(sorted(conditions)):
        result["condition_means"][condition] = {}
        for metric_index, metric in enumerate(METRICS):
            matrix = values[condition][metric]
            result["condition_means"][condition][metric] = {
                "mean": float(matrix.mean()),
                "crossed_patient_seed_bootstrap_95_ci": crossed_bootstrap(
                    matrix, seed=202670 + condition_index * 10 + metric_index
                ),
            }

    for metric_index, metric in enumerate(METRICS):
        delta = values["standard_penalty"][metric] - values["no_correction_penalty"][metric]
        result["paired_differences"][metric] = {
            "mean_delta": float(delta.mean()),
            "crossed_patient_seed_bootstrap_95_ci": crossed_bootstrap(
                delta, seed=202690 + metric_index
            ),
        }

    aggregate_output.mkdir(parents=True, exist_ok=False)
    summary_path = aggregate_output / "correction_penalty_aggregate.json"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    public_metadata = {
        "experiment_id": metadata["experiment_id"],
        "stage": metadata["stage"],
        "baseline_results_previously_inspected": metadata["baseline_results_previously_inspected"],
        "plan_sha256": metadata["plan_sha256"],
        "private_metadata_sha256": sha256(private_output / "metadata.json"),
        "private_patient_metrics_sha256": result["patient_level_csv_sha256"],
        "test_access": metadata["test_access"],
        "patient_level_rows_included": False,
    }
    (aggregate_output / "provenance.json").write_text(
        json.dumps(public_metadata, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
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
