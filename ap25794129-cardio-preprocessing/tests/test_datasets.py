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
    assert df.iloc[0]["phase"] == "diastole"
    assert int(df.iloc[0]["slice_index"]) == 1


def test_build_camus_manifest_from_temporary_dataset(tmp_path: Path):
    root = tmp_path / "camus"
    patient_dir = root / "database_nifti" / "patient0001"
    patient_dir.mkdir(parents=True)
    split_dir = root / "database_split"
    split_dir.mkdir(parents=True)
    (split_dir / "subgroup_training.txt").write_text("patient0001\n", encoding="utf-8")

    image = np.random.default_rng(1).random((16, 16), dtype=np.float32)
    mask = np.zeros((16, 16), dtype=np.float32)
    mask[4:10, 5:11] = 1.0
    _save_nifti(patient_dir / "patient0001_2CH_ED.nii.gz", image)
    _save_nifti(patient_dir / "patient0001_2CH_ED_gt.nii.gz", mask)
    _save_nifti(patient_dir / "patient0001_4CH_ES.nii.gz", image)
    _save_nifti(patient_dir / "patient0001_4CH_ES_gt.nii.gz", mask)

    df = build_camus_manifest(root)
    assert len(df) == 2
    assert set(df["subset"]) == {"training"}
    assert set(df["patient_id"]) == {"CAMUS_patient0001"}


def test_split_helpers_keep_patient_level_separation():
    df = np.array(
        [
            ("A_001", "training"),
            ("A_002", "training"),
            ("B_001", "validation"),
            ("C_001", "testing"),
        ],
        dtype=object,
    )
    import pandas as pd

    manifest = pd.DataFrame(df, columns=["patient_id", "subset"])
    manifest["phase"] = "diastole"
    manifest["image_path"] = "x.npy"
    manifest["mask_path"] = "y.npy"

    split_map = split_by_subset_column(manifest)
    assert set(split_map) == {"train", "val", "test"}

    random_split = make_patient_level_random_split(
        manifest,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=7,
    )
    combined_ids = set()
    for split_df in random_split.values():
        ids = set(split_df["patient_id"])
        assert not (combined_ids & ids)
        combined_ids |= ids
