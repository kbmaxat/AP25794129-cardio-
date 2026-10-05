"""Aggregate the exploratory resolution-augmentation parameter screen.

All reported contrasts are descriptive: the 32-patient sweep holdout is part of the
training partition and has prior research-use history. Official CAMUS dev/validation/
testing data are not read by this summarizer.
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
SEEDS = (2027, 2028, 2029, 2030, 2031, 2032)
CONFIGS = {
    "p025_2x": {"variant": "2x_plain", "probability": 0.25},
    "p075_2x": {"variant": "2x_plain", "probability": 0.75},
    "p050_4x": {"variant": "4x_plain", "probability": 0.50},
}
N_BOOTSTRAP = 50000


def read_patient_metrics(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected_variants = {"clean", "2x_plain", "4x_plain"}
    if len(rows) != 32 * 2 * len(expected_variants):
        raise ValueError(f"Unexpected row count in {path}: {len(rows)}")
    for row in rows:
        grouped.setdefault((row["patient"], row["variant"]), []).append(row)
    patients = sorted({patient for patient, _ in grouped})
    if len(patients) != 32:
        raise ValueError(f"Expected 32 sweep-evaluation patients in {path}, found {len(patients)}")
    result = {}
    for patient in patients:
        for variant in expected_variants:
            phases = grouped.get((patient, variant), [])
            if len(phases) != 2 or {row["phase"] for row in phases} != {"ED", "ES"}:
                raise ValueError(f"Expected ED and ES for {patient}/{variant} in {path}")
            result[(patient, variant)] = {
                metric: sum(float(row[metric]) for row in phases) / 2 for metric in METRICS
            }
    return result


def crossed_bootstrap(values: np.ndarray, seed: int) -> list[float]:
    if values.ndim != 2 or values.shape != (len(SEEDS), 32):
        raise ValueError(f"Expected 6x32 paired values, got {values.shape}")
    rng = np.random.default_rng(seed)
    estimates = np.empty(N_BOOTSTRAP, dtype=np.float64)
    for index in range(N_BOOTSTRAP):
        seed_indices = rng.integers(0, values.shape[0], values.shape[0])
        patient_indices = rng.integers(0, values.shape[1], values.shape[1])
        estimates[index] = values[np.ix_(seed_indices, patient_indices)].mean()
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p025-2x", type=Path, required=True)
    parser.add_argument("--p075-2x", type=Path, required=True)
    parser.add_argument("--p050-4x", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path, required=True)
    args = parser.parse_args()

    run_dirs = {
        "p025_2x": args.p025_2x.resolve(),
        "p075_2x": args.p075_2x.resolve(),
        "p050_4x": args.p050_4x.resolve(),
    }
    metadata = {}
    metrics = {}
    split_hashes = {}
    for name, run_dir in run_dirs.items():
        run_metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        if (
            run_metadata["stage"] != "exploratory_train_partition_augmentation_screening"
            or run_metadata.get("official_dev_accessed") is not False
            or run_metadata.get("test_access") is not False
            or run_metadata["seeds"] != list(SEEDS)
            or run_metadata["augmentation_variant"] != CONFIGS[name]["variant"]
            or run_metadata["augmentation_probability"] != CONFIGS[name]["probability"]
            or run_metadata["n_sweep_evaluation_patients"] != 32
        ):
            raise ValueError(f"Run metadata does not match predeclared screen for {name}")
        metadata[name] = run_metadata
        split_path = run_dir / "patient_subsplit.json"
        split_hashes[name] = sha256(split_path)
        metrics[name] = {}
        for seed in SEEDS:
            for arm in ("clean_baseline", "degradation_augmented"):
                key = (arm, seed)
                metrics[name][key] = read_patient_metrics(
                    run_dir / f"seed{seed}_{arm}" / "dev_phase_metrics.csv"
                )
        split = json.loads(split_path.read_text(encoding="utf-8"))
        if len(split["fit"]) != 256 or len(split["select"]) != 32 or len(split["sweep_evaluation"]) != 32:
            raise ValueError(f"Unexpected subsplit sizes for {name}")
        if set(split["sweep_evaluation"]) != set(
            patient for patient, _ in metrics[name][("clean_baseline", SEEDS[0])]
        ):
            raise ValueError(f"Evaluation patients differ from saved split for {name}")

    if len(set(split_hashes.values())) != 1:
        raise ValueError("Candidate runs do not share the identical patient subsplit")
    patient_ids = sorted(
        patient for patient, variant in metrics["p025_2x"][("clean_baseline", SEEDS[0])]
        if variant == "clean"
    )
    # Check common initialization/run determinism by ensuring duplicated baseline outcomes
    # agree across candidates before pooling the candidate-specific paired contrasts.
    for seed in SEEDS:
        baseline_reference = metrics["p025_2x"][("clean_baseline", seed)]
        for config in ("p075_2x", "p050_4x"):
            other = metrics[config][("clean_baseline", seed)]
            for key in baseline_reference:
                for metric in METRICS:
                    if not np.isclose(
                        baseline_reference[key][metric], other[key][metric], rtol=0, atol=1e-7
                    ):
                        raise ValueError(
                            f"Duplicated clean baseline differs across candidate runs for seed {seed}"
                        )

    def matrix(config: str, arm: str, variant: str, metric: str) -> np.ndarray:
        values = np.empty((len(SEEDS), len(patient_ids)), dtype=np.float64)
        for si, seed in enumerate(SEEDS):
            table = metrics[config][(arm, seed)]
            for pi, patient in enumerate(patient_ids):
                values[si, pi] = table[(patient, variant)][metric]
        return values

    result = {
        "experiment_id": "RESOLUTION-AUGMENTATION-PARAMETER-SCREEN-001",
        "stage": "exploratory_train_partition_augmentation_screening",
        "status_note": (
            "Descriptive parameter screen only; evaluation patients are train-partition "
            "patients with prior research-use history. Not independent or confirmatory."
        ),
        "n_seeds": len(SEEDS),
        "n_sweep_evaluation_patients": len(patient_ids),
        "official_dev_accessed": False,
        "test_access": False,
        "bootstrap_replicates": N_BOOTSTRAP,
        "protocol_sha256": metadata["p025_2x"]["protocol_sha256"],
        "patient_subsplit_sha256": split_hashes["p025_2x"],
        "configs": {},
    }
    for ci, (name, config) in enumerate(CONFIGS.items()):
        variant = config["variant"]
        output = {
            "augmentation_variant": variant,
            "augmentation_probability": config["probability"],
            "means": {},
            "contrasts": {},
        }
        for arm in ("clean_baseline", "degradation_augmented"):
            variants = ("clean", variant) if arm == "degradation_augmented" else ("clean", variant)
            output["means"][arm] = {
                v: {metric: float(matrix(name, arm, v, metric).mean()) for metric in METRICS}
                for v in set(variants)
            }
        contrasts = {
            "degradation_augmented_minus_clean_baseline_on_degraded": (
                ("degradation_augmented", variant), ("clean_baseline", variant),
            ),
            "clean_guardrail_augmented_minus_baseline": (
                ("degradation_augmented", "clean"), ("clean_baseline", "clean"),
            ),
            "descriptive_residual_gap_augmented_degraded_minus_baseline_clean": (
                ("degradation_augmented", variant), ("clean_baseline", "clean"),
            ),
            "baseline_degradation_loss_degraded_minus_clean": (
                ("clean_baseline", variant), ("clean_baseline", "clean"),
            ),
        }
        for contrast_index, (contrast_name, ((arm_a, var_a), (arm_b, var_b))) in enumerate(contrasts.items()):
            output["contrasts"][contrast_name] = {}
            for metric_index, metric in enumerate(METRICS):
                delta = matrix(name, arm_a, var_a, metric) - matrix(name, arm_b, var_b, metric)
                output["contrasts"][contrast_name][metric] = {
                    "mean_delta": float(delta.mean()),
                    "crossed_patient_seed_bootstrap_95_ci": crossed_bootstrap(
                        delta, seed=20261050 + ci * 100 + contrast_index * 10 + metric_index
                    ),
                }
        result["configs"][name] = output

    out_dir = args.aggregate_output.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "resolution_augmentation_parameter_screen_aggregate.json").write_text(
        json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    provenance = {
        "experiment_id": result["experiment_id"],
        "stage": result["stage"],
        "protocol_sha256": result["protocol_sha256"],
        "patient_subsplit_sha256": result["patient_subsplit_sha256"],
        "private_run_metadata_sha256": {
            name: sha256(run_dir / "metadata.json") for name, run_dir in run_dirs.items()
        },
        "private_phase_metrics_sha256": {
            f"{name}/seed{seed}_{arm}/dev_phase_metrics.csv": sha256(
                run_dir / f"seed{seed}_{arm}" / "dev_phase_metrics.csv"
            )
            for name, run_dir in run_dirs.items()
            for seed in SEEDS
            for arm in ("clean_baseline", "degradation_augmented")
        },
        "official_dev_accessed": False,
        "test_access": False,
        "patient_ids_included": False,
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    hashes = {p.name: sha256(p) for p in sorted(out_dir.iterdir()) if p.is_file()}
    (out_dir / "files_sha256.json").write_text(
        json.dumps(hashes, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=True, indent=2))
    print(f"OUTPUT={out_dir}")


if __name__ == "__main__":
    main()
