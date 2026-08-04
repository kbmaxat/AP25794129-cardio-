import pandas as pd
import pytest

from cardiac_image_system.core.metrics import dice_from_counts
from cardiac_image_system.core.validation import (
    aggregate_patient_level,
    aggregate_volumetric_level,
    validate_patient_level_split,
)


def test_validate_patient_level_split_raises_on_overlap():
    train = pd.DataFrame({"patient_id": ["p1", "p2"]})
    val = pd.DataFrame({"patient_id": ["p2"]})
    test = pd.DataFrame({"patient_id": ["p3"]})
    with pytest.raises(ValueError):
        validate_patient_level_split(train, val, test)


def test_aggregate_patient_level_means_per_slice_metric():
    df = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2"],
            "mode": ["none", "none", "none"],
            "dice": [1.0, 0.5, 0.8],
        }
    )
    result = aggregate_patient_level(df)
    p1_row = result[result["patient_id"] == "p1"].iloc[0]
    assert p1_row["dice"] == pytest.approx(0.75)


def test_aggregate_volumetric_level_pools_counts_not_ratios():
    # Two slices of the same (patient, phase, view) volume: one slice with a tiny true region
    # (huge relative error, would dominate a naive per-slice mean) and one large, near-perfect
    # slice. Pooled Dice should reflect the combined volume, not the mean of two very different
    # per-slice ratios.
    df = pd.DataFrame(
        {
            "patient_id": ["p1", "p1"],
            "phase": ["diastole", "diastole"],
            "view": ["SAX", "SAX"],
            "mode": ["none", "none"],
            "intersection_pixels": [0, 990],
            "foreground_pixels_true": [2, 1000],
            "foreground_pixels_pred": [0, 1000],
        }
    )
    result = aggregate_volumetric_level(df)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["n_slices"] == 2
    expected = dice_from_counts(intersection=990, true_area=1002, pred_area=1000)
    assert row["dice_3d"] == pytest.approx(expected)

    naive_mean_of_ratios = (0.0 + dice_from_counts(990, 1000, 1000)) / 2
    assert row["dice_3d"] != pytest.approx(naive_mean_of_ratios)


def test_aggregate_volumetric_level_groups_camus_views_separately():
    # CAMUS: one slice per (patient, phase, view); pooling must not mix 2CH and 4CH together.
    df = pd.DataFrame(
        {
            "patient_id": ["p1", "p1"],
            "phase": ["diastole", "diastole"],
            "view": ["2CH", "4CH"],
            "mode": ["none", "none"],
            "intersection_pixels": [80, 40],
            "foreground_pixels_true": [100, 100],
            "foreground_pixels_pred": [100, 50],
        }
    )
    result = aggregate_volumetric_level(df)
    assert len(result) == 2
    assert set(result["view"]) == {"2CH", "4CH"}


def test_aggregate_volumetric_level_requires_overlap_columns():
    df = pd.DataFrame({"patient_id": ["p1"], "dice": [1.0]})
    with pytest.raises(ValueError):
        aggregate_volumetric_level(df)
