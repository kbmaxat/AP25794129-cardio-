from __future__ import annotations

from pathlib import Path
import pandas as pd


def validate_patient_level_split(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    train_ids = set(train["patient_id"].astype(str))
    val_ids = set(val["patient_id"].astype(str))
    test_ids = set(test["patient_id"].astype(str))

    overlaps = {
        "train_val": train_ids & val_ids,
        "train_test": train_ids & test_ids,
        "val_test": val_ids & test_ids,
    }
    bad = {k: sorted(v) for k, v in overlaps.items() if v}
    if bad:
        raise ValueError(f"Patient-level leakage detected: {bad}")


def aggregate_patient_level(results: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = results.select_dtypes(include="number").columns.tolist()
    group_cols = [col for col in ["patient_id", "mode"] if col in results.columns]
    if not group_cols:
        raise ValueError("aggregate_patient_level requires at least a patient_id column")
    return results.groupby(group_cols, as_index=False)[numeric_cols].mean()


def aggregate_volumetric_level(results: pd.DataFrame) -> pd.DataFrame:
    """Pool per-slice overlap counts into a genuine volumetric (3D) Dice/IoU per anatomical volume.

    Groups by (patient_id, phase, view, mode) -- one group per reconstructed 3D volume (ACDC:
    one short-axis stack per patient per cardiac phase; CAMUS: one 2D frame per patient, phase,
    and view, so pooling is a no-op there) -- and sums each group's intersection/true-area/
    pred-area pixel counts before computing a single ratio, rather than averaging per-slice
    ratios. This is not the same statistic as ``aggregate_patient_level``'s mean-of-slice-Dice,
    and is the statistically correct way to report a multi-slice volume's Dice/IoU.
    """
    from cardiac_image_system.core.metrics import dice_from_counts, iou_from_counts

    required = {"patient_id", "intersection_pixels", "foreground_pixels_true", "foreground_pixels_pred"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"aggregate_volumetric_level requires columns {sorted(missing)}")

    group_cols = [col for col in ["patient_id", "phase", "view", "mode"] if col in results.columns]
    pooled = results.groupby(group_cols, as_index=False).agg(
        intersection_pixels=("intersection_pixels", "sum"),
        foreground_pixels_true=("foreground_pixels_true", "sum"),
        foreground_pixels_pred=("foreground_pixels_pred", "sum"),
        n_slices=("patient_id", "size"),
    )
    pooled["dice_3d"] = pooled.apply(
        lambda r: dice_from_counts(r["intersection_pixels"], r["foreground_pixels_true"], r["foreground_pixels_pred"]),
        axis=1,
    )
    pooled["iou_3d"] = pooled.apply(
        lambda r: iou_from_counts(r["intersection_pixels"], r["foreground_pixels_true"], r["foreground_pixels_pred"]),
        axis=1,
    )
    return pooled


def save_runtime_log(output_dir: str | Path, metadata: dict) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "runtime_log.txt"
    with path.open("a", encoding="utf-8") as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
