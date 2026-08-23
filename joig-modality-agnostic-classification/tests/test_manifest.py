from pathlib import Path

import pandas as pd
import pytest

from joig_cardio.data import assign_grouped_folds, load_manifest


def test_manifest_and_grouped_folds_do_not_leak_patients(tmp_path: Path):
    rows = []
    for patient in range(10):
        for frame in range(2):
            rows.append({"path": f"missing/{patient}_{frame}.png", "label": patient % 2,
                         "patient_id": f"P{patient:02d}", "modality": "echo" if patient < 5 else "mri", "dataset": "test"})
    path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    result = assign_grouped_folds(load_manifest(path, check_files=False), folds=5)
    assert result.groupby("patient_id")["fold"].nunique().max() == 1
    assert set(result.fold) == set(range(5))


def test_manifest_rejects_nonbinary_label(tmp_path: Path):
    path = tmp_path / "manifest.csv"
    pd.DataFrame([{"path": "x.png", "label": 2, "patient_id": "P1", "modality": "echo", "dataset": "test"}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="binary"):
        load_manifest(path, check_files=False)
