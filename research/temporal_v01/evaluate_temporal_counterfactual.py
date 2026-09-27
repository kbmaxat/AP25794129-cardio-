from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PACKAGE = ROOT / "ap25794129-cardio-preprocessing"
sys.path.insert(0, str(PACKAGE))

import numpy as np
import torch
from torch.utils.data import DataLoader

from cardiac_image_system.research_temporal.data import load_cases, read_splits, sha256
from cardiac_image_system.research_temporal.model import TemporalSystem
from run_pilot import (
    boundary_distances,
    collate_samples,
    native_prediction,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dev-only temporal input counterfactual analysis.")
    parser.add_argument("--run", type=Path, required=True, help="Completed factual temporal training run")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run = args.run.resolve()
    if json.loads((run / "status.json").read_text(encoding="utf-8"))["status"] != "completed":
        raise ValueError("Only completed training runs can be evaluated")
    config = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    if config.get("eligible_partition") != "training" or config.get("external_test_access") is not False:
        raise ValueError("Counterfactual analysis requires a training-only checkpoint protocol")
    if config.get("geometry_mode", "squash") != "squash":
        raise ValueError("This protocol requires the locked squash geometry")
    if config.get("confidence_ablation", "measured") != "measured":
        raise ValueError("Counterfactual source checkpoint must have been trained with measured confidence")

    split = json.loads((run / "patient_split.json").read_text(encoding="utf-8"))
    official_splits = read_splits(args.data_root)
    dev_patients = split["dev"]
    if len(dev_patients) != config["dev_patients"] or not set(dev_patients) <= set(official_splits["training"]):
        raise ValueError("Run dev patients do not match the training-only protocol")

    data, _ = load_cases(args.data_root, dev_patients, config)
    loader = DataLoader(
        data,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_samples,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = run / "temporal" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint["mode"] != "temporal":
        raise ValueError("Checkpoint must be the temporal-factual model")
    model = TemporalSystem("temporal", config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    rows = []
    with torch.no_grad():
        for batch in loader:
            model_inputs = [
                batch[key].to(device)
                for key in ("raw", "aligned", "confidence", "time_offsets")
            ]
            logits_factual, image_factual, change_factual = model(*model_inputs)
            logits_factual_repeat, image_factual_repeat, _ = model(*model_inputs)
            neutral_inputs = [
                model_inputs[0],
                model_inputs[1],
                torch.zeros_like(model_inputs[2]),
                model_inputs[3],
            ]
            logits_neutral, image_neutral, change_neutral = model(*neutral_inputs)
            prob_factual = logits_factual.sigmoid().cpu().numpy()[:, 0]
            prob_neutral = logits_neutral.sigmoid().cpu().numpy()[:, 0]
            image_delta = (image_factual - image_neutral).abs().cpu().numpy()[:, 0]
            prob_delta = np.abs(prob_factual - prob_neutral)
            image_repeat_delta = (image_factual - image_factual_repeat).abs().cpu().numpy()[:, 0]
            prob_repeat_delta = np.abs(
                prob_factual - logits_factual_repeat.sigmoid().cpu().numpy()[:, 0]
            )
            correction_factual = change_factual.abs().cpu().numpy()[:, 0]
            correction_neutral = change_neutral.abs().cpu().numpy()[:, 0]
            for index in range(len(batch["patient"])):
                target = np.asarray(batch["native_mask"][index], dtype=bool)
                mode_metrics = {}
                for mode, probability in (
                    ("factual", prob_factual),
                    ("neutralized", prob_neutral),
                ):
                    prediction = native_prediction(probability, batch, index) >= 0.5
                    intersection = np.logical_and(prediction, target).sum()
                    dice = (2 * intersection + 1e-6) / (
                        prediction.sum() + target.sum() + 1e-6
                    )
                    area_error = abs(prediction.sum() - target.sum()) / max(target.sum(), 1)
                    mean_surface, hausdorff = boundary_distances(prediction, target)
                    mode_metrics[mode] = {
                        "dice": float(dice),
                        "relative_area_error": float(area_error),
                        "mean_surface_distance_px": mean_surface,
                        "hausdorff_distance_px": hausdorff,
                    }
                rows.append({
                    "patient": batch["patient"][index],
                    "phase": batch["phase"][index],
                    **{
                        f"{mode}_{metric}": value
                        for mode, metrics in mode_metrics.items()
                        for metric, value in metrics.items()
                    },
                    "processed_image_mae": float(image_delta[index].mean()),
                    "probability_map_mae": float(prob_delta[index].mean()),
                    "processed_image_repeat_mae": float(image_repeat_delta[index].mean()),
                    "probability_map_repeat_mae": float(prob_repeat_delta[index].mean()),
                    "factual_mean_abs_correction": float(correction_factual[index].mean()),
                    "neutralized_mean_abs_correction": float(correction_neutral[index].mean()),
                })

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "dev_counterfactual_phase_metrics.csv", rows)
    patient_rows = []
    fields = [key for key in rows[0] if key not in ("patient", "phase")]
    patients = sorted({row["patient"] for row in rows})
    for patient in patients:
        patient_rows.append({
            "patient": patient,
            **{
                key: float(np.mean([float(row[key]) for row in rows if row["patient"] == patient]))
                for key in fields
            },
        })
    write_csv(output / "dev_counterfactual_patient_metrics.csv", patient_rows)
    metadata = {
        "stage": "development_exploratory",
        "dataset": "CAMUS",
        "partition": "official_training_internal_dev",
        "n_dev_patients": len(patients),
        "n_dev_phases": len(rows),
        "checkpoint_run": run.name,
        "checkpoint_sha256": sha256(checkpoint_path),
        "training_protocol_sha256": sha256(run / "protocol.json"),
        "dev_split_sha256": sha256(run / "patient_split.json"),
        "test_access": False,
        "same_checkpoint_factual_vs_neutralized": True,
        "device": device,
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
