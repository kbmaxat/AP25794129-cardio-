from pathlib import Path

import nibabel as nib
import numpy as np

from cardiac_image_system.core.io import get_nifti_in_plane_spacing, load_grayscale_image


def test_load_nifti_slice(tmp_path: Path):
    arr = np.zeros((8, 8, 3), dtype=np.float32)
    arr[:, :, 1] = 5.0
    path = tmp_path / "sample.nii.gz"
    nib.save(nib.Nifti1Image(arr, np.eye(4)), path)

    loaded = load_grayscale_image(path, slice_index=1)
    assert loaded.shape == (8, 8)
    assert np.isfinite(loaded).all()
    assert loaded.max() == 1.0


def test_load_nifti_without_slice_uses_middle_slice(tmp_path: Path):
    arr = np.zeros((6, 6, 5), dtype=np.float32)
    arr[:, :, 2] = 3.0
    path = tmp_path / "sample_middle.nii.gz"
    nib.save(nib.Nifti1Image(arr, np.eye(4)), path)

    loaded = load_grayscale_image(path)
    assert loaded.shape == (6, 6)
    assert loaded.max() == 1.0


def test_get_nifti_in_plane_spacing_reads_header_zooms(tmp_path: Path):
    arr = np.zeros((8, 8, 3), dtype=np.float32)
    image = nib.Nifti1Image(arr, np.eye(4))
    image.header.set_zooms((1.5, 2.0, 10.0))
    path = tmp_path / "spaced.nii.gz"
    nib.save(image, path)

    spacing = get_nifti_in_plane_spacing(path)
    assert spacing == (1.5, 2.0)


def test_get_nifti_in_plane_spacing_returns_none_for_non_nifti(tmp_path: Path):
    path = tmp_path / "sample.npy"
    np.save(path, np.zeros((4, 4), dtype=np.float32))
    assert get_nifti_in_plane_spacing(path) is None
