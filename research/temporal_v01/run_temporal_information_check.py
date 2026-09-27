from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PACKAGE = ROOT / "ap25794129-cardio-preprocessing"
sys.path.insert(0, str(PACKAGE))

import numpy as np

from cardiac_image_system.research_temporal.data import sha256
from export_pilot import csv_rows, read_json, write_csv, write_json


METRICS = (
    "dice",
    "relative_area_error",
    "mean_surface_distance_px",
    "hausdorff_distance_px",
)
ARMS = ("none", "spatial", "temporal")


def patient_means(path: Path) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in csv_rows(path):
        grouped[row["patient"]].append(row)
    if not grouped:
        raise ValueError(f"No patient rows in {path}")
    return {
        patient: {
            metric: float(np.mean([float(row[metric]) for row in rows]))
            for metric in METRICS
        }
        for patient, rows in grouped.items()
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


def run_command(command: list[str], log_path: Path, cwd: Path) -> str:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"Command failed ({completed.returncode}); see {log_path}")
    return completed.stdout


def run_directory(stdout: str) -> Path:
    paths = [
        line.split("=", 1)[1]
        for line in stdout.splitlines()
        if line.startswith("RUN_DIRECTORY=")
    ]
    if len(paths) != 1:
        raise ValueError("Expected exactly one RUN_DIRECTORY in runner output")
    return Path(paths[0]).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run approved temporal-information development study.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--plan",
        type=Path,
        default=HERE / "temporal_information_plan_v04.json",
    )
    args = parser.parse_args()
    plan = read_json(args.plan)
    if plan["test_access"] is not False or plan["hypothesis_tests"] is not False:
        raise ValueError("This runner only supports training/dev exploratory analysis")
    if plan["arms"] != ["temporal_factual", "temporal_neutral", "spatial", "identity"]:
        raise ValueError("Unexpected locked arm set")

    protocol_path = HERE / "protocol_v04_temporal_information_candidate.md"
    stamp = datetime.now(timezone.utc).strftime("temporal_information_v04_%Y%m%dT%H%M%SZ")
    private = HERE / "private" / stamp
    private.mkdir(parents=True, exist_ok=False)
    write_json(private / "plan.json", plan)
    shutil.copy2(protocol_path, private / protocol_path.name)
    shutil.copy2(Path(__file__), private / Path(__file__).name)
    write_json(private / "status.json", {
        "status": "running",
        "utc": datetime.now(timezone.utc).isoformat(),
        "test_access": False,
    })

    base_command = [
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
    ]
    run_records = {}
    provenance = {}
    paired_patient_values: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    counterfactual_rows = []

    try:
        for seed in plan["training_seeds"]:
            factual_command = base_command + [
                "--training-seed",
                str(seed),
                "--modes",
                *ARMS,
            ]
            factual_stdout = run_command(
                factual_command,
                private / f"seed_{seed}_factual.log",
                ROOT,
            )
            factual_run = run_directory(factual_stdout)

            neutral_command = base_command + [
                "--training-seed",
                str(seed),
                "--modes",
                "temporal",
                "--confidence-ablation",
                "zeros",
            ]
            neutral_stdout = run_command(
                neutral_command,
                private / f"seed_{seed}_neutral.log",
                ROOT,
            )
            neutral_run = run_directory(neutral_stdout)

            factual_config = read_json(factual_run / "protocol.json")
            neutral_config = read_json(neutral_run / "protocol.json")
            if factual_config["training_seed"] != seed or neutral_config["training_seed"] != seed:
                raise ValueError(f"Seed mismatch for {seed}")
            if neutral_config.get("confidence_ablation") != "zeros":
                raise ValueError(f"Neutral arm not configured for zero confidence: seed {seed}")
            if factual_config.get("geometry_mode") != "squash":
                raise ValueError(f"Factual arm did not use locked squash geometry: seed {seed}")

            factual_split_hash = sha256(factual_run / "patient_split.json")
            neutral_split_hash = sha256(neutral_run / "patient_split.json")
            if factual_split_hash != neutral_split_hash:
                raise ValueError(f"Patient split mismatch for seed {seed}")
            factual_temporal = read_json(factual_run / "temporal" / "summary.json")
            neutral_temporal = read_json(neutral_run / "temporal" / "summary.json")
            if factual_temporal["initial_model_sha256"] != neutral_temporal["initial_model_sha256"]:
                raise ValueError(f"Temporal model initialization mismatch for seed {seed}")

            run_records[seed] = {
                "factual": factual_run,
                "neutral": neutral_run,
                "factual_summary": factual_run / "summary.csv",
                "neutral_summary": neutral_run / "summary.csv",
            }

            for arm in ARMS:
                mode_path = factual_run / arm
                if not mode_path.exists():
                    raise ValueError(f"Missing factual arm {arm} for seed {seed}")
                summary_rows = csv_rows(factual_run / "summary.csv")
                summary = next(row for row in summary_rows if row["mode"] == arm)
                run_records[seed][arm] = {
                    "run": factual_run,
                    "summary": summary,
                    "patient_metrics": patient_means(mode_path / "dev_phase_metrics.csv"),
                }
            run_records[seed]["temporal_neutral"] = {
                "run": neutral_run,
                "summary": next(row for row in csv_rows(neutral_run / "summary.csv") if row["mode"] == "temporal"),
                "patient_metrics": patient_means(neutral_run / "temporal" / "dev_phase_metrics.csv"),
            }

            factual_patients = set(run_records[seed]["temporal"]["patient_metrics"])
            neutral_patients = set(run_records[seed]["temporal_neutral"]["patient_metrics"])
            if len(factual_patients) != plan["dev_patients"] or factual_patients != neutral_patients:
                raise ValueError(f"Dev patient mismatch or count error for seed {seed}")
            if any(
                set(run_records[seed][arm]["patient_metrics"]) != factual_patients
                for arm in ARMS
            ):
                raise ValueError(f"Reference-arm dev patients differ for seed {seed}")
            paired_patient_values[seed] = {
                arm: run_records[seed][arm]["patient_metrics"] for arm in (*ARMS, "temporal_neutral")
            }

            counterfactual_dir = private / f"seed_{seed}_counterfactual"
            counterfactual_command = [
                sys.executable,
                str(HERE / "evaluate_temporal_counterfactual.py"),
                "--run",
                str(factual_run),
                "--data-root",
                str(args.data_root.resolve()),
                "--output",
                str(counterfactual_dir),
            ]
            run_command(
                counterfactual_command,
                private / f"seed_{seed}_counterfactual.log",
                ROOT,
            )
            counterfactual_rows.extend({
                "seed": seed,
                **row,
            } for row in csv_rows(counterfactual_dir / "dev_counterfactual_patient_metrics.csv"))

            for label, run in (("factual", factual_run), ("neutral", neutral_run)):
                provenance[f"{seed}_{label}"] = {
                    "run_id": run.name,
                    "protocol_sha256": sha256(run / "protocol.json"),
                    "patient_split_sha256": sha256(run / "patient_split.json"),
                    "initial_temporal_model_sha256": read_json(run / "temporal" / "summary.json")["initial_model_sha256"],
                    "temporal_checkpoint_sha256": sha256(run / "temporal" / "best.pt"),
                    "temporal_dev_metrics_sha256": sha256(run / "temporal" / "dev_phase_metrics.csv"),
                }
            write_json(private / "provenance_progress.json", provenance)
            write_json(private / "status.json", {
                "status": "running",
                "completed_seeds": [int(item) for item in run_records],
                "utc": datetime.now(timezone.utc).isoformat(),
                "test_access": False,
            })
            print(json.dumps({
                "seed": seed,
                "factual_run": factual_run.name,
                "neutral_run": neutral_run.name,
                "n_dev_patients": len(factual_patients),
            }), flush=True)

        seeds = list(plan["training_seeds"])
        first_seed_patients = sorted(paired_patient_values[seeds[0]]["temporal"])
        if any(
            sorted(paired_patient_values[seed]["temporal"]) != first_seed_patients
            for seed in seeds
        ):
            raise ValueError("Dev patient split differs across training seeds")

        contrasts = {
            "temporal_factual_minus_temporal_neutral": ("temporal", "temporal_neutral"),
            "temporal_factual_minus_spatial": ("temporal", "spatial"),
            "temporal_factual_minus_identity": ("temporal", "none"),
        }
        patient_seed_differences = []
        seed_differences = []
        contrast_arrays: dict[str, dict[str, np.ndarray]] = {}
        for contrast, (left, right) in contrasts.items():
            contrast_arrays[contrast] = {
                metric: np.empty((len(seeds), len(first_seed_patients)), dtype=np.float64)
                for metric in METRICS
            }
            for seed_index, seed in enumerate(seeds):
                for patient_index, patient in enumerate(first_seed_patients):
                    lvalues = paired_patient_values[seed][left][patient]
                    rvalues = paired_patient_values[seed][right][patient]
                    diffs = {metric: lvalues[metric] - rvalues[metric] for metric in METRICS}
                    patient_seed_differences.append({
                        "seed": seed,
                        "patient": patient,
                        "contrast": contrast,
                        **diffs,
                    })
                    for metric, value in diffs.items():
                        contrast_arrays[contrast][metric][seed_index, patient_index] = value
                seed_differences.append({
                    "seed": seed,
                    "contrast": contrast,
                    **{
                        metric: float(contrast_arrays[contrast][metric][seed_index].mean())
                        for metric in METRICS
                    },
                })

        arm_seed_rows = []
        for seed in seeds:
            for arm_key in (*ARMS, "temporal_neutral"):
                record = run_records[seed][arm_key]
                summary = record["summary"]
                patient_metrics = record["patient_metrics"]
                arm_seed_rows.append({
                    "seed": seed,
                    "arm": "identity" if arm_key == "none" else arm_key,
                    "run_id": record["run"].name,
                    "n_dev_patients": len(patient_metrics),
                    "epochs": plan["epochs"],
                    "best_epoch": int(summary["best_epoch"]),
                    **{
                        metric: float(np.mean([patient_metrics[p][metric] for p in patient_metrics]))
                        for metric in METRICS
                    },
                })

        public = HERE / "public_results" / stamp
        public.mkdir(parents=True, exist_ok=False)
        write_json(public / "plan.json", plan)
        write_csv(public / "arm_seed_metrics.csv", arm_seed_rows)
        write_csv(public / "paired_seed_differences.csv", seed_differences)

        aggregate = {
            "stage": "development_exploratory",
            "n_unique_dev_patients": len(first_seed_patients),
            "n_training_seeds": len(seeds),
            "test_access": False,
            "p_values_computed": False,
            "delta_definition": "left arm minus right arm",
            "contrasts": {},
        }
        for contrast, metric_arrays in contrast_arrays.items():
            aggregate["contrasts"][contrast] = {}
            for metric, values in metric_arrays.items():
                aggregate["contrasts"][contrast][metric] = {
                    "mean_delta": float(values.mean()),
                    "sd_across_seed_means_ddof1": float(values.mean(axis=1).std(ddof=1)),
                    "crossed_patient_seed_bootstrap_95_ci": crossed_bootstrap(
                        values,
                        seed=20260927 + len(aggregate["contrasts"]) * 101 + len(aggregate["contrasts"][contrast]),
                    ),
                }

        cf_fields = [key for key in counterfactual_rows[0] if key not in ("seed", "patient", "phase")]
        cf_patients = sorted({row["patient"] for row in counterfactual_rows})
        counterfactual_aggregate = {}
        for field in cf_fields:
            values = np.empty((len(seeds), len(cf_patients)), dtype=np.float64)
            for seed_index, seed in enumerate(seeds):
                seed_rows = {
                    row["patient"]: float(row[field])
                    for row in counterfactual_rows
                    if row["seed"] == seed
                }
                if set(seed_rows) != set(cf_patients):
                    raise ValueError(f"Counterfactual patient set differs for seed {seed}")
                values[seed_index] = [seed_rows[patient] for patient in cf_patients]
            counterfactual_aggregate[field] = {
                "mean": float(values.mean()),
                "crossed_patient_seed_bootstrap_95_ci": crossed_bootstrap(
                    values,
                    seed=202619 + len(counterfactual_aggregate),
                ),
            }
        aggregate["same_checkpoint_counterfactual_mean_over_patient_phase_seed"] = counterfactual_aggregate
        write_json(public / "aggregate.json", aggregate)
        write_json(public / "provenance.json", provenance)
        write_json(public / "files_sha256.json", {
            path.name: sha256(path)
            for path in sorted(public.iterdir())
            if path.is_file() and path.name != "files_sha256.json"
        })
        write_csv(private / "patient_seed_differences.csv", patient_seed_differences)
        write_csv(private / "counterfactual_patient_metrics.csv", counterfactual_rows)
        write_json(private / "status.json", {
            "status": "complete",
            "public_directory": public.name,
            "utc": datetime.now(timezone.utc).isoformat(),
            "test_access": False,
        })
        print(f"PUBLIC_RESULTS={public}", flush=True)
        print(json.dumps(aggregate, ensure_ascii=True), flush=True)
    except Exception as error:
        write_json(private / "status.json", {
            "status": "failed",
            "error": repr(error),
            "completed_seeds": [int(item) for item in run_records],
            "utc": datetime.now(timezone.utc).isoformat(),
            "test_access": False,
        })
        raise


if __name__ == "__main__":
    main()
