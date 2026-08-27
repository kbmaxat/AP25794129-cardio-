import pandas as pd
import pytest

from cardiac_image_system.core.splits import (
    make_patient_level_random_split,
    split_by_subset_column,
    split_by_subset_column_carving_validation,
)


def _acdc_like_manifest(n_train: int = 20, n_test: int = 10) -> pd.DataFrame:
    rows = []
    for i in range(n_train):
        rows.append({"patient_id": f"ACDC_patient{i:03d}", "subset": "training"})
    for i in range(n_train, n_train + n_test):
        rows.append({"patient_id": f"ACDC_patient{i:03d}", "subset": "testing"})
    return pd.DataFrame(rows)


def test_carving_validation_leaves_test_set_untouched_across_validation_seeds():
    df = _acdc_like_manifest()
    split_a = split_by_subset_column_carving_validation(df, val_ratio=0.15, validation_seed=1)
    split_b = split_by_subset_column_carving_validation(df, val_ratio=0.15, validation_seed=2)

    test_ids_a = set(split_a["test"]["patient_id"])
    test_ids_b = set(split_b["test"]["patient_id"])
    assert test_ids_a == test_ids_b
    assert test_ids_a == {f"ACDC_patient{i:03d}" for i in range(20, 30)}


def test_carving_validation_partitions_training_pool_without_leakage():
    df = _acdc_like_manifest()
    split_map = split_by_subset_column_carving_validation(df, val_ratio=0.2, validation_seed=42)

    train_ids = set(split_map["train"]["patient_id"])
    val_ids = set(split_map["val"]["patient_id"])
    test_ids = set(split_map["test"]["patient_id"])

    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)
    assert train_ids | val_ids == {f"ACDC_patient{i:03d}" for i in range(20)}
    assert len(val_ids) == 4  # 20 * 0.2


def test_carving_validation_delegates_when_validation_tag_already_present():
    df = _acdc_like_manifest()
    df = pd.concat(
        [df, pd.DataFrame([{"patient_id": "ACDC_patient099", "subset": "validation"}])],
        ignore_index=True,
    )
    delegated = split_by_subset_column_carving_validation(df, val_ratio=0.15, validation_seed=1)
    direct = split_by_subset_column(df)
    assert set(delegated["val"]["patient_id"]) == set(direct["val"]["patient_id"])
    assert set(delegated["train"]["patient_id"]) == set(direct["train"]["patient_id"])


def test_carving_validation_raises_without_test_patients():
    df = _acdc_like_manifest(n_train=20, n_test=0)
    with pytest.raises(ValueError):
        split_by_subset_column_carving_validation(df)


def _mixed_corpus_like_manifest() -> pd.DataFrame:
    rows = []
    for i in range(20):
        rows.append({"patient_id": f"ACDC_patient{i:03d}", "subset": "training", "dataset": "ACDC"})
    for i in range(20, 30):
        rows.append({"patient_id": f"ACDC_patient{i:03d}", "subset": "testing", "dataset": "ACDC"})
    for i in range(30):
        subset = "validation" if i < 5 else ("testing" if i >= 25 else "training")
        rows.append({"patient_id": f"CAMUS_patient{i:03d}", "subset": subset, "dataset": "CAMUS"})
    return pd.DataFrame(rows)


def test_carving_validation_carves_only_the_dataset_missing_native_validation():
    # ACDC has no native validation tag; CAMUS does. A naive "val is non-empty, done" check
    # would silently produce a CAMUS-only validation set with zero ACDC patients, biasing
    # checkpoint selection for any run trained on the pooled manifest. The fix must carve
    # validation for ACDC specifically while leaving CAMUS's native validation set untouched.
    df = _mixed_corpus_like_manifest()
    split_map = split_by_subset_column_carving_validation(
        df, val_ratio=0.2, validation_seed=42, stratify_by=["dataset"]
    )

    val_datasets = set(split_map["val"]["dataset"])
    assert val_datasets == {"ACDC", "CAMUS"}, f"validation set missing a dataset: {val_datasets}"

    train_ids = set(split_map["train"]["patient_id"])
    val_ids = set(split_map["val"]["patient_id"])
    test_ids = set(split_map["test"]["patient_id"])
    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)

    # CAMUS's native validation patients (000-004) must be preserved untouched.
    assert {f"CAMUS_patient{i:03d}" for i in range(5)} <= val_ids
    # ACDC must have gained validation patients carved from its own 20-patient training pool.
    acdc_val = [pid for pid in val_ids if pid.startswith("ACDC_")]
    assert len(acdc_val) == 4  # 20 * 0.2


def test_carving_validation_matches_random_split_train_val_boundary_behavior():
    # Sanity check that the carve step reuses make_patient_level_random_split's own logic
    # (same seed, same ratio) rather than diverging in size rounding.
    df = _acdc_like_manifest(n_train=20, n_test=10)
    carved = split_by_subset_column_carving_validation(df, val_ratio=0.15, validation_seed=7)
    manual = make_patient_level_random_split(
        df[df["subset"] == "training"], train_ratio=0.85, val_ratio=0.15, test_ratio=0.0, seed=7
    )
    assert set(carved["train"]["patient_id"]) == set(manual["train"]["patient_id"])
    assert set(carved["val"]["patient_id"]) == set(manual["val"]["patient_id"])
