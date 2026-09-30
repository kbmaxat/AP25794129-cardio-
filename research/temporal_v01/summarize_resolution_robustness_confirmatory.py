"""Aggregate the resolution-robustness confirmatory experiment: patient-level (ED/ES
average) dev metrics per seed/arm/variant, the three pre-registered contrasts, and
patient-and-seed crossed percentile bootstrap 95% CIs.

Registered protocol: resolution_robustness_confirmatory_protocol_v01.md
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

METRICS = ("dice", "relative_area_error", "mean_surface_distance_px", "hausdorff_distance_px")
ARMS = ("clean_baseline", "degradation_augmented")
VARIANTS = ("clean", "2x_plain", "4x_plain")
SEEDS = (2027, 2028, 2029, 2030, 2031, 2032)

# (arm, variant) minus (arm, variant), as pre-registered in the protocol.
CONTRASTS = {
    "replication_check_baseline_2x_minus_clean": (
        ("clean_baseline", "2x_plain"), ("clean_baseline", "clean"),
    ),
    "primary_augmented_minus_baseline_on_2x": (
        ("degradation_augmented", "2x_plain"), ("clean_baseline", "2x_plain"),
    ),
    "guardrail_augmented_minus_baseline_on_clean": (
        ("degradation_augmented", "clean"), ("clean_baseline", "clean"),
    ),
    # Descriptive only, not part of the decision rule (protocol section "Degradation used...").
    "descriptive_augmented_minus_baseline_on_4x": (
        ("degradation_augmented", "4x_plain"), ("clean_baseline", "4x_plain"),
    ),
    # Added post-hoc after independent review (see protocol "Post-hoc correction"): no
    # threshold was pre-registered for this contrast. It answers "how much of the original
    # clean-vs-degraded gap remains after augmentation, in absolute terms" directly, instead
    # of only comparing within-arm or within-variant differences. Descriptive only.
    "descriptive_residual_gap_augmented_2x_minus_baseline_clean": (
        ("degradation_augmented", "2x_plain"), ("clean_baseline", "clean"),
    ),
}


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


def patient_level(rows: list[dict]) -> dict[tuple[str, int], dict[str, float]]:
    """Average ED/ES per (patient, variant) for one arm/seed's dev_phase_metrics.csv rows."""
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["patient"], row["variant"]), []).append(row)
    result = {}
    for (patient, variant), phases in grouped.items():
        if len(phases) != 2:
            raise ValueError(f"Expected exactly 2 phases for {patient}/{variant}, got {len(phases)}")
        result[(patient, variant)] = {
            metric: sum(float(row[metric]) for row in phases) / len(phases) for metric in METRICS
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path, required=True)
    args = parser.parse_args()

    private_output = args.private_output.resolve()
    aggregate_output = args.aggregate_output.resolve()
    metadata = json.loads((private_output / "metadata.json").read_text(encoding="utf-8"))
    seeds = tuple(metadata["seeds"])
    if seeds != SEEDS:
        raise ValueError(f"Unexpected seeds in metadata: {seeds}")

    per_arm_seed: dict[tuple[str, int], dict[tuple[str, str], dict[str, float]]] = {}
    patients_by_arm_seed: dict[tuple[str, int], list[str]] = {}
    for arm in ARMS:
        for seed in seeds:
            csv_path = private_output / f"seed{seed}_{arm}" / "dev_phase_metrics.csv"
            with csv_path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            per_arm_seed[(arm, seed)] = patient_level(rows)
            patients_by_arm_seed[(arm, seed)] = sorted({p for p, _ in per_arm_seed[(arm, seed)]})

    reference_patients = patients_by_arm_seed[(ARMS[0], seeds[0])]
    for key, patients in patients_by_arm_seed.items():
        if patients != reference_patients:
            raise ValueError(f"Dev patient set mismatch for {key}")
    patients = reference_patients
    if len(patients) != metadata["n_dev_patients"]:
        raise ValueError("Dev patient count does not match run metadata")

    def matrix(arm: str, variant: str, metric: str) -> np.ndarray:
        out = np.empty((len(seeds), len(patients)), dtype=np.float64)
        for seed_index, seed in enumerate(seeds):
            table = per_arm_seed[(arm, seed)]
            for patient_index, patient in enumerate(patients):
                out[seed_index, patient_index] = table[(patient, variant)][metric]
        return out

    result = {
        "experiment_id": "RESOLUTION-ROBUSTNESS-CONFIRMATORY-001",
        "stage": "confirmatory",
        "n_seeds": len(seeds),
        "n_dev_patients": len(patients),
        "test_access": metadata["test_access"],
        "protocol_sha256": metadata["protocol_sha256"],
        "arm_variant_means": {},
        "contrasts": {},
    }
    for arm in ARMS:
        result["arm_variant_means"][arm] = {}
        for variant in VARIANTS:
            result["arm_variant_means"][arm][variant] = {
                metric: float(matrix(arm, variant, metric).mean()) for metric in METRICS
            }

    for contrast_index, (name, ((arm_a, variant_a), (arm_b, variant_b))) in enumerate(CONTRASTS.items()):
        result["contrasts"][name] = {}
        for metric_index, metric in enumerate(METRICS):
            delta = matrix(arm_a, variant_a, metric) - matrix(arm_b, variant_b, metric)
            result["contrasts"][name][metric] = {
                "mean_delta": float(delta.mean()),
                "crossed_patient_seed_bootstrap_95_ci": crossed_bootstrap(
                    delta, seed=930000 + contrast_index * 10 + metric_index
                ),
            }

    aggregate_output.mkdir(parents=True, exist_ok=False)
    summary_path = aggregate_output / "resolution_robustness_confirmatory_aggregate.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    per_seed_csv_hashes = {
        f"seed{seed}_{arm}/dev_phase_metrics.csv": sha256(
            private_output / f"seed{seed}_{arm}" / "dev_phase_metrics.csv"
        )
        for arm in ARMS for seed in seeds
    }
    public_metadata = {
        "experiment_id": result["experiment_id"],
        "stage": result["stage"],
        "protocol_sha256": metadata["protocol_sha256"],
        "private_metadata_sha256": sha256(private_output / "metadata.json"),
        "private_subsplit_sha256": sha256(private_output / "patient_subsplit.json"),
        "per_seed_dev_metrics_sha256": per_seed_csv_hashes,
        "test_access": metadata["test_access"],
        "patient_level_rows_included": False,
    }
    (aggregate_output / "provenance.json").write_text(
        json.dumps(public_metadata, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    hashes = {
        path.name: sha256(path) for path in sorted(aggregate_output.iterdir()) if path.is_file()
    }
    (aggregate_output / "files_sha256.json").write_text(
        json.dumps(hashes, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=True, indent=2))
    print(f"OUTPUT={aggregate_output}")


if __name__ == "__main__":
    main()
