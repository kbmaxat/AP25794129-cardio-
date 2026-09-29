from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PACKAGE = ROOT / "ap25794129-cardio-preprocessing"
sys.path.insert(0, str(PACKAGE))

from cardiac_image_system.research_temporal.data import sha256
from export_pilot import csv_rows, read_json, write_json


METRICS = (
    "dice",
    "relative_area_error",
    "mean_surface_distance_px",
    "hausdorff_distance_px",
    "mean_abs_correction",
    "max_abs_correction",
)
LOCKED_CONFIG_KEYS = (
    "eligible_partition",
    "view",
    "phases",
    "target_label",
    "train_patients",
    "dev_patients",
    "split_seed",
    "image_size",
    "batch_size",
    "epochs",
    "unet_base_channels",
    "adapter_channels",
    "epsilon",
    "optimizer",
    "learning_rate",
    "boundary_weight",
    "augmentation",
    "checkpoint_rule",
    "precision",
    "torch_threads",
    "external_test_access",
    "neighbor_policy",
    "normalization",
    "alignment",
    "geometry_mode",
    "confidence_ablation",
)


def patient_means(path: Path) -> dict[str, dict[str, float]]:
    # Despite the function name (kept for compatibility with existing call
    # sites/aggregates), most metrics are the ED/ES phase average per patient.
    # max_abs_correction is the exception: it takes the per-patient maximum
    # across phases, matching its "largest single correction" semantics.
    # See correction_penalty_protocol_v01.md "Outcomes" for the documented
    # aggregation rule per metric.
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in csv_rows(path):
        grouped.setdefault(row["patient"], []).append(row)
    if not grouped:
        raise ValueError(f"No patient metrics in {path}")
    return {
        patient: {
            metric: (
                max(float(row[metric]) for row in phases)
                if metric == "max_abs_correction"
                else sum(float(row[metric]) for row in phases) / len(phases)
            )
            for metric in METRICS
        }
        for patient, phases in grouped.items()
    }


def run_pilot(command: list[str], log_path: Path) -> Path:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Training failed ({result.returncode}); inspect {log_path}")
    run_paths = [
        line.split("=", 1)[1]
        for line in result.stdout.splitlines()
        if line.startswith("RUN_DIRECTORY=")
    ]
    if len(run_paths) != 1:
        raise ValueError(f"Expected one RUN_DIRECTORY line in {log_path}")
    return Path(run_paths[0]).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train matched zero-correction-penalty arms against existing v0.4 references."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baseline-provenance", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=HERE / "correction_penalty_plan_v01.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan_path = args.plan.resolve()
    plan = read_json(plan_path)
    provenance_path = args.baseline_provenance.resolve()
    baseline_provenance = read_json(provenance_path)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)

    paired_rows = []
    run_records = {}
    for seed in plan["training_seeds"]:
        key = f"{seed}_factual"
        if key not in baseline_provenance:
            raise ValueError(f"Missing baseline provenance for seed {seed}")
        baseline_run = HERE / "runs" / baseline_provenance[key]["run_id"]
        if read_json(baseline_run / "status.json")["status"] != "completed":
            raise ValueError(f"Baseline run is not complete: {baseline_run}")
        baseline_config = read_json(baseline_run / "protocol.json")
        baseline_split_hash = sha256(baseline_run / "patient_split.json")
        baseline_summary = read_json(baseline_run / "temporal" / "summary.json")
        if baseline_config.get("correction_weight") != 0.05:
            raise ValueError(f"Expected reference correction_weight=0.05 for seed {seed}")
        if baseline_config.get("training_seed") != seed:
            raise ValueError(f"Baseline training seed mismatch for seed {seed}")
        if baseline_split_hash != baseline_provenance[key]["patient_split_sha256"]:
            raise ValueError(f"Baseline split hash mismatch for seed {seed}")
        if sha256(baseline_run / "temporal" / "best.pt") != baseline_provenance[key]["temporal_checkpoint_sha256"]:
            raise ValueError(f"Baseline checkpoint hash mismatch for seed {seed}")

        command = [
            sys.executable,
            str(HERE / "run_pilot.py"),
            "--data-root",
            str(args.data_root.resolve()),
            "--train-patients",
            str(plan["train_patients"]),
            "--dev-patients",
            str(plan["dev_patients"]),
            "--epochs",
            str(plan["epochs"]),
            "--geometry-mode",
            "squash",
            "--training-seed",
            str(seed),
            "--modes",
            "temporal",
            "--correction-weight",
            "0.0",
        ]
        no_penalty_run = run_pilot(command, output / f"seed_{seed}_no_penalty.log")
        no_penalty_config = read_json(no_penalty_run / "protocol.json")
        if no_penalty_config.get("correction_weight") != 0.0:
            raise ValueError(f"No-penalty run has wrong correction weight for seed {seed}")
        if no_penalty_config.get("training_seed") != seed:
            raise ValueError(f"No-penalty run seed mismatch for seed {seed}")
        if any(
            no_penalty_config.get(name) != baseline_config.get(name)
            for name in LOCKED_CONFIG_KEYS
        ):
            raise ValueError(f"Locked training settings differ for seed {seed}")
        no_penalty_split_hash = sha256(no_penalty_run / "patient_split.json")
        if no_penalty_split_hash != baseline_split_hash:
            raise ValueError(f"Patient split mismatch for seed {seed}")
        no_penalty_summary = read_json(no_penalty_run / "temporal" / "summary.json")
        if no_penalty_summary["initial_model_sha256"] != baseline_summary["initial_model_sha256"]:
            raise ValueError(f"Initial model mismatch for seed {seed}")

        condition_values = {
            "standard_penalty": patient_means(
                baseline_run / "temporal" / "dev_phase_metrics.csv"
            ),
            "no_correction_penalty": patient_means(
                no_penalty_run / "temporal" / "dev_phase_metrics.csv"
            ),
        }
        expected_patients = set(condition_values["standard_penalty"])
        if len(expected_patients) != plan["dev_patients"] or set(
            condition_values["no_correction_penalty"]
        ) != expected_patients:
            raise ValueError(f"Patient-level metrics do not pair for seed {seed}")
        for condition, patients in condition_values.items():
            for patient, metrics in patients.items():
                paired_rows.append({
                    "seed": seed,
                    "patient": patient,
                    "condition": condition,
                    **metrics,
                })
        run_records[str(seed)] = {
            "baseline_run_id": baseline_run.name,
            "baseline_checkpoint_sha256": sha256(baseline_run / "temporal" / "best.pt"),
            "baseline_initial_model_sha256": baseline_summary["initial_model_sha256"],
            "baseline_split_sha256": baseline_split_hash,
            "no_penalty_run_id": no_penalty_run.name,
            "no_penalty_checkpoint_sha256": sha256(no_penalty_run / "temporal" / "best.pt"),
            "no_penalty_initial_model_sha256": no_penalty_summary["initial_model_sha256"],
            "no_penalty_split_sha256": no_penalty_split_hash,
        }
        write_json(output / "run_records.json", run_records)
        print(json.dumps({
            "seed": seed,
            "baseline_run": baseline_run.name,
            "no_penalty_run": no_penalty_run.name,
            "n_dev_patients": len(expected_patients),
        }), flush=True)

    if not paired_rows:
        raise ValueError("No paired results were produced")
    csv_path = output / "correction_penalty_patient_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    metadata = {
        "experiment_id": plan["experiment_id"],
        "stage": plan["stage"],
        "baseline_results_previously_inspected": plan["baseline_results_previously_inspected"],
        "dataset": "CAMUS",
        "partition": "official_training_internal_dev",
        "n_train_patients": plan["train_patients"],
        "n_dev_patients": plan["dev_patients"],
        "n_seeds": len(plan["training_seeds"]),
        "test_access": False,
        "p_values_computed": False,
        "plan_sha256": sha256(plan_path),
        "baseline_provenance_sha256": sha256(provenance_path),
        "run_records": run_records,
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=True))
    print(f"OUTPUT={output}")


if __name__ == "__main__":
    main()
