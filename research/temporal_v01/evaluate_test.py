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

import torch
from torch.utils.data import DataLoader

from cardiac_image_system.research_temporal.data import load_cases, read_splits, sha256
from cardiac_image_system.research_temporal.model import TemporalSystem
from run_pilot import evaluate

MODES = ("none", "direct_temporal", "spatial", "temporal")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed pilot checkpoints on official CAMUS testing.")
    parser.add_argument("run", type=Path, help="Completed pilot run containing mode/best.pt checkpoints")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence-ablation", choices=("measured", "ones", "zeros"))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    args = parser.parse_args()

    run = args.run.resolve()
    if json.loads((run / "status.json").read_text(encoding="utf-8"))["status"] != "completed":
        raise ValueError("Only completed runs can be evaluated")
    config = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    if args.confidence_ablation is not None:
        config["confidence_ablation"] = args.confidence_ablation
    if config["eligible_partition"] != "training" or config["external_test_access"]:
        raise ValueError("Checkpoint protocol must have been trained without test access")

    splits = read_splits(args.data_root)
    test_patients = splits["testing"]
    test_data, test_records = load_cases(args.data_root, test_patients, config)
    if config.get("confidence_ablation") == "ones":
        for sample in test_data:
            sample["confidence"] = torch.ones_like(torch.from_numpy(sample["confidence"])).numpy()
    elif config.get("confidence_ablation") == "zeros":
        for sample in test_data:
            sample["confidence"] = torch.zeros_like(torch.from_numpy(sample["confidence"])).numpy()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "test_data_manifest.json").write_text(
        json.dumps(test_records, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    summaries = []
    all_rows = []
    for mode in args.modes:
        checkpoint_path = run / mode / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model = TemporalSystem(mode, config)
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        loader = DataLoader(test_data, batch_size=config["batch_size"], shuffle=False, num_workers=0)
        loss, rows, patients = evaluate(model, loader, config, device)
        rows = [{"mode": mode, **row} for row in rows]
        all_rows.extend(rows)
        summaries.append({
            "mode": mode,
            "checkpoint_sha256": sha256(checkpoint_path),
            "checkpoint_epoch": checkpoint["epoch"],
            "test_loss": loss,
            "test_patient_dice": sum(patients.values()) / len(patients),
            "test_n_patients": len(patients),
            "test_n_phases": len(rows),
        })
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_csv(output / "test_phase_metrics.csv", all_rows)
    write_csv(output / "test_summary.csv", summaries)
    (output / "metadata.json").write_text(json.dumps({
        "stage": "fixed_checkpoint_internal_holdout_exploratory",
        "dataset": "CAMUS",
        "partition": "official_testing",
        "training_run": run.name,
        "training_protocol_sha256": sha256(run / "protocol.json"),
        "testing_split_sha256": sha256(args.data_root / "database_split" / "subgroup_testing.txt"),
        "checkpoint_selection": "preselected_by_training_dev_loss",
        "same_dataset_holdout": True,
        "confidence_ablation": config.get("confidence_ablation", "measured"),
        "evaluated_modes": args.modes,
        "evaluation_config": {key: value for key, value in config.items() if key != "dataset_root"},
        "evaluation_script_sha256": sha256(Path(__file__)),
        "device": device,
    }, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
