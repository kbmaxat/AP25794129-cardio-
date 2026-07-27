from pathlib import Path

import nibabel as nib
import numpy as np

from cardiac_image_system.core.datasets import build_acdc_manifest, build_camus_manifest
from cardiac_image_system.core.splits import make_patient_level_random_split, split_by_subset_column


def _save_nifti(path: Path, array: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(array.astype(np.float32), np.eye(4)), path)


def test_build_acdc_manifest_from_temporary_dataset(tmp_path: Path):
    root = tmp_path / "acdc" / "training" / "patient001"
    root.mkdir(parents=True)
    (root / "Info.cfg").write_text("ED: 1\nES: 12\nGroup: NOR\n", encoding="utf-8")
    image = np.random.default_rng(0).random((8, 8, 2), dtype=np.float32)
    mask = np.zeros((8, 8, 2), dtype=np.float32)
    mask[2:6, 2:6, 1] = 1.0
    _save_nifti(root / "patient001_frame01.nii.gz", image)
    _save_nifti(root / "patient001_frame01_gt.nii.gz", mask)
    df = build_acdc_manifest(tmp_path / "acdc", subsets=("training",), include_empty_masks=False)
    assert len(df) == 1
    assert df.iloc[0]["patient_id"] == "ACDC_patient001"


def test_split_helpers_keep_patient_level_separation():
    import pandas as pd
    df = np.array([("A_001", "training"), ("B_001", "validation"), ("C_001", "testing")], dtype=object)
    manifest = pd.DataFrame(df, columns=["patient_id", "subset"])
    manifest["phase"] = "diastole"
    manifest["image_path"] = "x.npy"
    manifest["mask_path"] = "y.npy"
    split_map = split_by_subset_column(manifest)
    assert set(split_map) == {"train", "val", "test"}
    random_split = make_patient_level_random_split(manifest, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=7)
    combined_ids = set()
    for split_df in random_split.values():
        ids = set(split_df["patient_id"])
        assert not (combined_ids & ids)
        combined_ids |= ids
