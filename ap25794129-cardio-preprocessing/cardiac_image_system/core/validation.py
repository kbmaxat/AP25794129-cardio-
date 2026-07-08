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


def save_runtime_log(output_dir: str | Path, metadata: dict) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "runtime_log.txt"
    with path.open("a", encoding="utf-8") as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
