from pathlib import Path

import nibabel as nib
import numpy as np

from cardiac_image_system.core.io import load_grayscale_image


def test_load_nifti_slice(tmp_path: Path):
    arr = np.zeros((8, 8, 3), dtype=np.float32)
    arr[:, :, 1] = 5.0
    path = tmp_path / "sample.nii.gz"
    nib.save(nib.Nifti1Image(arr, np.eye(4)), path)
    loaded = load_grayscale_image(path, slice_index=1)
    assert loaded.shape == (8, 8)
    assert loaded.max() == 1.0
