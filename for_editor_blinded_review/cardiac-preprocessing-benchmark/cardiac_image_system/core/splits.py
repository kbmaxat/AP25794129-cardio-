from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cardiac_image_system.core.validation import validate_patient_level_split


def split_by_subset_column(df: pd.DataFrame, subset_col: str = "subset") -> dict[str, pd.DataFrame]:
    if subset_col not in df.columns:
        raise ValueError(f"Column '{subset_col}' not found in manifest")
    normalized = df[subset_col].astype(str).str.lower()
    train = df[normalized.isin(["training", "train"])].copy()
    val = df[normalized.isin(["validation", "val"])].copy()
    test = df[normalized.isin(["testing", "test"])].copy()
    validate_patient_level_split(train, val, test)
    return {"train": train, "val": val, "test": test}


def make_patient_level_random_split(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.1, test_ratio: float = 0.2, seed: int = 42, stratify_by: list[str] | None = None) -> dict[str, pd.DataFrame]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")
    stratify_by = stratify_by or []
    patient_cols = ["patient_id", *[col for col in stratify_by if col in df.columns]]
    patient_df = df[patient_cols].drop_duplicates("patient_id").reset_index(drop=True)
    if patient_df.empty:
        return {"train": df.iloc[0:0].copy(), "val": df.iloc[0:0].copy(), "test": df.iloc[0:0].copy()}
    rng = np.random.default_rng(seed)
    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []
    if stratify_by:
        grouped = patient_df.groupby(stratify_by, dropna=False)
        patient_groups = [group for _, group in grouped]
    else:
        patient_groups = [patient_df]
    for patient_group in patient_groups:
        ids = patient_group["patient_id"].astype(str).to_numpy()
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_train = min(n_train, n)
        n_val = min(n_val, max(0, n - n_train))
        n_test_start = n_train + n_val
        train_ids.extend(ids[:n_train].tolist())
        val_ids.extend(ids[n_train:n_test_start].tolist())
        test_ids.extend(ids[n_test_start:].tolist())
    train = df[df["patient_id"].astype(str).isin(train_ids)].copy()
    val = df[df["patient_id"].astype(str).isin(val_ids)].copy()
    test = df[df["patient_id"].astype(str).isin(test_ids)].copy()
    validate_patient_level_split(train, val, test)
    return {"train": train, "val": val, "test": test}


def export_split_manifests(split_map: dict[str, pd.DataFrame], output_dir: str | Path, prefix: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_df in split_map.items():
        split_df.to_csv(output_dir / f"{prefix}_{split_name}.csv", index=False)
