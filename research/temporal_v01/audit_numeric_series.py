from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from export_pilot import csv_rows, read_json, sha256, write_csv, write_json

HERE = Path(__file__).resolve().parent
EXPECTED_SEEDS = [2026, 2027, 2028, 2029, 2030]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=HERE / "runs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, provenance, reference = [], {}, None
    for path in sorted(args.runs.glob("pilot_*/protocol.json")):
        config = read_json(path)
        if config["epochs"] != 50 or config["modes"] != ["none", "spatial"]:
            continue
        run = path.parent
        if read_json(run / "status.json")["status"] != "completed":
            raise ValueError(f"Incomplete eligible run: {run.name}")
        comparison = {key: value for key, value in config.items() if key not in {"training_seed", "dataset_root"}}
        signature = (comparison, sha256(run / "patient_split.json"))
        if reference is None:
            reference = signature
        elif reference != signature:
            raise ValueError("Configurations or patient split differ across seeds")
        values = {r["mode"]: float(r["dev_patient_dice"]) for r in csv_rows(run / "summary.csv")}
        rows.append({"run_id": run.name, "training_seed": config["training_seed"], "epochs": 50,
                     "partition": "dev", "n_patients": config["dev_patients"],
                     "none_dice": values["none"], "spatial_dice": values["spatial"],
                     "spatial_minus_none": values["spatial"] - values["none"]})
        provenance[run.name] = {"protocol_sha256": sha256(path), "summary_sha256": sha256(run / "summary.csv"),
                                "patient_split_sha256": sha256(run / "patient_split.json")}
    rows.sort(key=lambda row: row["training_seed"])
    if [r["training_seed"] for r in rows] != EXPECTED_SEEDS:
        raise ValueError("Expected exactly one completed paired run per specified seed")
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "spatial_50epoch_seed_metrics.csv", rows)
    diffs = [row["spatial_minus_none"] for row in rows]
    write_json(args.output / "spatial_50epoch_aggregate.json", {
        "stage": "development_exploratory", "partition": "dev", "n_seeds": len(rows),
        "n_unique_dev_patients": rows[0]["n_patients"], "n_splits": 1,
        "mean_spatial_minus_none": statistics.mean(diffs), "sd_difference_ddof1": statistics.stdev(diffs),
        "positive_seed_differences": sum(value > 0 for value in diffs),
        "min_difference": min(diffs), "max_difference": max(diffs), "hypothesis_testing": False,
    })
    test_rows = []
    for folder in sorted(args.runs.glob("test_eval*")):
        if not (folder / "test_summary.csv").exists():
            continue
        metadata = read_json(folder / "metadata.json")
        for row in csv_rows(folder / "test_summary.csv"):
            test_rows.append({"evaluation_id": folder.name, "training_run": metadata["training_run"],
                              "mode": row["mode"], "partition": "CAMUS_internal_testing",
                              "checkpoint_epoch": row["checkpoint_epoch"], "dice": row["test_patient_dice"],
                              "n_patients": row["test_n_patients"], "checkpoint_sha256": row["checkpoint_sha256"],
                              "confidence_ablation_recorded": "confidence_ablation" in metadata,
                              "confidence_ablation": metadata.get("confidence_ablation", "not_recorded")})
        provenance[folder.name] = {"summary_sha256": sha256(folder / "test_summary.csv"),
                                   "metadata_sha256": sha256(folder / "metadata.json")}
    write_csv(args.output / "existing_holdout_metrics.csv", test_rows)
    write_json(args.output / "source_hashes.json", provenance)
    write_json(args.output / "files_sha256.json", {p.name: sha256(p) for p in sorted(args.output.iterdir())})
    print(json.dumps({"seeds": len(rows), "mean_difference": statistics.mean(diffs), "sd_ddof1": statistics.stdev(diffs)}))


if __name__ == "__main__":
    main()
