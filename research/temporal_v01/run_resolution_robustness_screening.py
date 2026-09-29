"""Frozen-model screening: does controlled resolution reduction hurt segmentation, and
does a single fixed (non-learned) restoration step recover it?

Registered protocol: resolution_robustness_screening_protocol_v01.md
No training happens in this script. The segmenter checkpoint is loaded once and frozen.
Only the 320 official CAMUS training-partition "train" patients (per the reused split in
runs/pilot_20260927T093824Z_f0cd02/patient_split.json) are used. Masks are never modified.
"""
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

import cv2
import nibabel as nib
import numpy as np
import torch

from cardiac_image_system.research_temporal.data import sha256
from cardiac_image_system.research_temporal.model import TemporalSystem

CHECKPOINT_RUN = HERE / "runs" / "pilot_20260927T093824Z_f0cd02"
EXPECTED_CHECKPOINT_SHA256 = "8aecb07cf37970b39a9bce7de52942aab933d8ce15748fb0192d39ebdd4212fe"
VIEW = "4CH"
PHASES = ("ED", "ES")
TARGET_LABEL = 1
IMAGE_SIZE = 128
VARIANTS = ("clean", "2x_plain", "2x_sharpened", "4x_plain", "4x_sharpened")
METRICS = (
    "dice",
    "relative_area_error",
    "mean_surface_distance_px",
    "hausdorff_distance_px",
    "spurious_components",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def boundary_distances(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Symmetric mean and Hausdorff boundary distances in native pixels (matches run_pilot.py)."""
    kernel = np.ones((3, 3), dtype=np.uint8)
    pred_boundary = cv2.morphologyEx(prediction.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    target_boundary = cv2.morphologyEx(target.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    if not target_boundary.any():
        raise ValueError("Boundary metric requires a non-empty target")
    if not pred_boundary.any():
        penalty = float(np.hypot(*prediction.shape))
        return penalty, penalty
    pred_distance = cv2.distanceTransform((~pred_boundary).astype(np.uint8), cv2.DIST_L2, 3)
    target_distance = cv2.distanceTransform((~target_boundary).astype(np.uint8), cv2.DIST_L2, 3)
    forward = target_distance[pred_boundary]
    backward = pred_distance[target_boundary]
    distances = np.concatenate([forward, backward])
    return float(distances.mean()), float(distances.max())


def spurious_components(prediction: np.ndarray) -> int:
    count, _ = cv2.connectedComponents(prediction.astype(np.uint8))
    return max(count - 1 - 1, 0)  # subtract background label, then the one expected blob


def make_variant(clean_128: np.ndarray, variant: str) -> np.ndarray:
    if variant == "clean":
        return clean_128
    factor, restoration = variant.split("_")
    small_size = IMAGE_SIZE // {"2x": 2, "4x": 4}[factor]
    small = cv2.resize(clean_128, (small_size, small_size), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(small, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    if restoration == "sharpened":
        blurred = cv2.GaussianBlur(restored, ksize=(0, 0), sigmaX=1.0)
        restored = np.clip(restored + 1.0 * (restored - blurred), 0.0, 1.0)
    return restored.astype(np.float32)


def load_phase_image(root: Path, patient: str, phase: str) -> tuple[np.ndarray, np.ndarray]:
    folder = root / "database_nifti" / patient
    image_path = folder / f"{patient}_{VIEW}_{phase}.nii.gz"
    mask_path = folder / f"{patient}_{VIEW}_{phase}_gt.nii.gz"
    image = np.asarray(nib.load(image_path).dataobj, dtype=np.float32).squeeze()
    mask = np.asarray(nib.load(mask_path).dataobj).squeeze()
    if mask.shape != image.shape:
        raise ValueError(f"Geometry mismatch {patient}/{phase}")
    if not np.isin(mask, [0, 1, 2, 3]).all():
        raise ValueError(f"Unknown mask labels {patient}/{phase}")
    target = (mask == TARGET_LABEL).astype(np.float32)
    if not target.any():
        raise ValueError(f"Empty target {patient}/{phase}")
    lo, hi = np.percentile(image, [1, 99])
    if hi <= lo:
        raise ValueError(f"Constant image {patient}/{phase}")
    normalized = np.clip((image - lo) / (hi - lo), 0, 1).astype(np.float32)
    return normalized, target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint_path = CHECKPOINT_RUN / "none" / "best.pt"
    actual_sha256 = sha256(checkpoint_path)
    if actual_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            f"Checkpoint hash mismatch: expected {EXPECTED_CHECKPOINT_SHA256}, got {actual_sha256}"
        )
    patient_split = json.loads((CHECKPOINT_RUN / "patient_split.json").read_text(encoding="utf-8"))
    train_patients = sorted(patient_split["train"])
    if len(train_patients) != 320:
        raise ValueError(f"Expected 320 train patients, found {len(train_patients)}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = TemporalSystem("none", checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.eval()

    data_root = args.data_root.resolve()
    private_output = args.private_output.resolve()
    private_output.mkdir(parents=True, exist_ok=False)

    phase_rows = []
    with torch.no_grad():
        for patient in train_patients:
            for phase in PHASES:
                normalized, native_target = load_phase_image(data_root, patient, phase)
                clean_128 = cv2.resize(
                    normalized, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA
                ).astype(np.float32)
                for variant in VARIANTS:
                    image_128 = make_variant(clean_128, variant)
                    tensor = torch.from_numpy(image_128)[None, None, :, :]
                    logits = model.segmenter(tensor)
                    probability = logits[0, 0].sigmoid().numpy()
                    native_size = (native_target.shape[1], native_target.shape[0])
                    native_probability = cv2.resize(
                        probability, native_size, interpolation=cv2.INTER_LINEAR
                    )
                    native_pred = native_probability >= 0.5
                    native_mask = native_target.astype(bool)
                    intersection = np.logical_and(native_pred, native_mask).sum()
                    dice = (2 * intersection + 1e-6) / (
                        native_pred.sum() + native_mask.sum() + 1e-6
                    )
                    area_error = abs(native_pred.sum() - native_mask.sum()) / max(
                        native_mask.sum(), 1
                    )
                    mean_surface_distance, hausdorff_distance = boundary_distances(
                        native_pred, native_mask
                    )
                    phase_rows.append({
                        "patient": patient, "phase": phase, "variant": variant,
                        "dice": float(dice), "relative_area_error": float(area_error),
                        "mean_surface_distance_px": mean_surface_distance,
                        "hausdorff_distance_px": hausdorff_distance,
                        "spurious_components": spurious_components(native_pred),
                    })
            print(f"done patient={patient}", flush=True)

    write_csv(private_output / "resolution_robustness_phase_metrics.csv", phase_rows)
    metadata = {
        "experiment_id": "RESOLUTION-ROBUSTNESS-SCREENING-001",
        "stage": "technical_screening_frozen_model_not_confirmatory",
        "utc_started_is_approximate": now(),
        "checkpoint_run": CHECKPOINT_RUN.name,
        "checkpoint_sha256": actual_sha256,
        "protocol_sha256": sha256(HERE / "resolution_robustness_screening_protocol_v01.md"),
        "n_train_patients": len(train_patients),
        "phases": list(PHASES),
        "variants": list(VARIANTS),
        "test_access": False,
        "hypothesis_tests": False,
    }
    (private_output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"OUTPUT={private_output}")


if __name__ == "__main__":
    main()
