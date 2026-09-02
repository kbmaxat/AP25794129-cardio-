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


def split_by_subset_column_carving_validation(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    validation_seed: int = 42,
    stratify_by: list[str] | None = None,
    subset_col: str = "subset",
) -> dict[str, pd.DataFrame]:
    """Like ``split_by_subset_column``, but tolerates a manifest whose subset column has no
    ``validation`` tag (e.g. ACDC's native training/testing-only labeling).

    The ``test`` split is exactly the subset-tagged test patients and is therefore identical
    across every caller regardless of the caller's own training seed. Validation is carved out
    of the subset-tagged training patients using ``validation_seed``, a constant independent of
    the training run's own seed, so that different training seeds see the same train/val/test
    patient composition and differ only in model initialization and optimization, not in which
    patients they are evaluated against. If the manifest already provides a non-empty
    ``validation`` tag, this delegates to ``split_by_subset_column`` unchanged.

    When ``df`` contains a ``dataset`` column spanning more than one dataset (e.g. a pooled
    ACDC+CAMUS manifest), a native ``validation`` tag supplied by only some of those datasets
    (CAMUS provides one; ACDC's own subset labeling does not) is *not* treated as sufficient:
    validation is additionally carved, per dataset, for exactly the datasets missing native
    validation coverage, so the returned validation set -- and therefore checkpoint selection --
    is never silently restricted to a subset of the datasets actually being trained on.
    """
    if subset_col not in df.columns:
        raise ValueError(f"Column '{subset_col}' not found in manifest")

    normalized = df[subset_col].astype(str).str.lower()
    train_pool = df[normalized.isin(["training", "train"])].copy()
    val = df[normalized.isin(["validation", "val"])].copy()
    test = df[normalized.isin(["testing", "test"])].copy()

    if train_pool.empty or test.empty:
        raise ValueError("Subset column must provide non-empty training and testing patients")

    if "dataset" in df.columns:
        train_pool_datasets = set(train_pool["dataset"].astype(str).unique())
        val_datasets = set(val["dataset"].astype(str).unique()) if not val.empty else set()
        missing_datasets = train_pool_datasets - val_datasets
    else:
        missing_datasets = set() if not val.empty else {None}

    if not missing_datasets:
        validate_patient_level_split(train_pool, val, test)
        return {"train": train_pool, "val": val, "test": test}

    if "dataset" in df.columns:
        is_missing = train_pool["dataset"].astype(str).isin(missing_datasets)
        carve_pool = train_pool[is_missing].copy()
        keep_pool = train_pool[~is_missing].copy()
    else:
        carve_pool = train_pool
        keep_pool = train_pool.iloc[0:0].copy()

    carved = make_patient_level_random_split(
        carve_pool,
        train_ratio=1.0 - val_ratio,
        val_ratio=val_ratio,
        test_ratio=0.0,
        seed=validation_seed,
        stratify_by=stratify_by,
    )
    train = pd.concat([keep_pool, carved["train"]], ignore_index=True)
    val = pd.concat([val, carved["val"]], ignore_index=True) if not val.empty else carved["val"]
    validate_patient_level_split(train, val, test)
    return {"train": train, "val": val, "test": test}


def make_patient_level_random_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
    stratify_by: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
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
