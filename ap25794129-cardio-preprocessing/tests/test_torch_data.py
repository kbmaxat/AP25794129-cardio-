from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from cardiac_image_system.core.preprocessing import MmSpaceFilterTargets, PreprocessParams, preprocess_image
from cardiac_image_system.core.torch_data import ManifestSegmentationDataset, cached_preprocess_image


def test_manifest_segmentation_dataset_returns_expected_shapes(tmp_path: Path):
    image = np.zeros((32, 24), dtype=np.uint8)
    image[8:24, 6:18] = 200
    mask = np.zeros((32, 24), dtype=np.uint8)
    mask[10:20, 8:16] = 255

    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask).save(mask_path)

    manifest = pd.DataFrame(
        [
            {
                "patient_id": "p001",
                "phase": "diastole",
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "dataset": "synthetic",
                "subset": "train",
            }
        ]
    )

    dataset = ManifestSegmentationDataset(manifest, image_size=(64, 64), preprocess_mode="none", augment=False)
    sample = dataset[0]
    assert sample["image"].shape == (1, 64, 64)
    assert sample["mask"].shape == (1, 64, 64)
    assert float(sample["mask"].max()) == 1.0
    assert np.isnan(sample["resized_spacing_row_mm"])
    assert np.isnan(sample["resized_spacing_col_mm"])


def test_manifest_segmentation_dataset_scales_spacing_by_resize_ratio(tmp_path: Path):
    image = np.zeros((32, 16), dtype=np.uint8)
    image[8:24, 4:12] = 200
    mask = np.zeros((32, 16), dtype=np.uint8)
    mask[10:20, 5:10] = 255

    image_path = tmp_path / "image_spaced.png"
    mask_path = tmp_path / "mask_spaced.png"
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask).save(mask_path)

    manifest = pd.DataFrame(
        [
            {
                "patient_id": "p003",
                "phase": "diastole",
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "dataset": "synthetic",
                "subset": "train",
                "spacing_row_mm": 1.5,
                "spacing_col_mm": 2.0,
            }
        ]
    )

    # native (32, 16) resized to (64, 64): row spacing halves (32/64), col spacing quarters (16/64)
    dataset = ManifestSegmentationDataset(manifest, image_size=(64, 64), preprocess_mode="none", augment=False)
    sample = dataset[0]
    assert sample["resized_spacing_row_mm"] == pytest.approx(1.5 * (32 / 64))
    assert sample["resized_spacing_col_mm"] == pytest.approx(2.0 * (16 / 64))


def test_manifest_segmentation_dataset_returns_multiclass_targets(tmp_path: Path):
    image = np.zeros((20, 18), dtype=np.uint8)
    image[4:16, 5:13] = 180
    mask = np.zeros((20, 18), dtype=np.uint8)
    mask[3:8, 3:8] = 1
    mask[8:14, 8:14] = 2
    mask[10:18, 2:6] = 3

    image_path = tmp_path / "image_mc.png"
    mask_path = tmp_path / "mask_mc.png"
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask).save(mask_path)

    manifest = pd.DataFrame(
        [
            {
                "patient_id": "p002",
                "phase": "systole",
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "dataset": "synthetic",
                "subset": "train",
            }
        ]
    )

    dataset = ManifestSegmentationDataset(
        manifest,
        image_size=(32, 32),
        preprocess_mode="none",
        augment=False,
        label_mode="multiclass",
        class_values=(0, 1, 2, 3),
    )
    sample = dataset[0]
    assert sample["image"].shape == (1, 32, 32)
    assert sample["mask"].shape == (4, 32, 32)
    assert sample["mask_labels"].shape == (32, 32)
    assert int(sample["mask_labels"].max()) == 3
    assert np.allclose(sample["mask"].sum(dim=0).numpy(), 1.0)


def test_cached_preprocess_image_matches_uncached_and_persists(tmp_path: Path):
    rng = np.random.default_rng(0)
    image = rng.random((16, 16), dtype=np.float32)
    cache_dir = tmp_path / "cache"
    params = PreprocessParams()

    direct = preprocess_image(image, mode="wavelet", params=params)
    cached_first = cached_preprocess_image(
        image, cache_dir=cache_dir, image_path="img.nii.gz", slice_index=0, mode="wavelet", params=params
    )
    assert np.allclose(direct, cached_first)

    cache_files = list((cache_dir / "wavelet").glob("*.npy"))
    assert len(cache_files) == 1

    # Second call with identical key must hit the cache: modifying the on-disk array and
    # calling again should return the (stale) cached array, proving preprocess_image was not
    # re-invoked, rather than silently recomputing and overwriting it.
    tampered = np.zeros_like(direct)
    np.save(cache_files[0], tampered)
    cached_second = cached_preprocess_image(
        image, cache_dir=cache_dir, image_path="img.nii.gz", slice_index=0, mode="wavelet", params=params
    )
    assert np.allclose(cached_second, tampered)


def test_cached_preprocess_image_distinguishes_mode_and_params(tmp_path: Path):
    rng = np.random.default_rng(1)
    image = rng.random((16, 16), dtype=np.float32)
    cache_dir = tmp_path / "cache"

    cached_preprocess_image(
        image, cache_dir=cache_dir, image_path="a.nii.gz", slice_index=0, mode="nlm", params=PreprocessParams()
    )
    cached_preprocess_image(
        image,
        cache_dir=cache_dir,
        image_path="a.nii.gz",
        slice_index=0,
        mode="nlm",
        params=PreprocessParams(nlm_h_multiplier=1.6),
    )
    cached_preprocess_image(
        image, cache_dir=cache_dir, image_path="b.nii.gz", slice_index=0, mode="nlm", params=PreprocessParams()
    )

    cache_files = list((cache_dir / "nlm").glob("*.npy"))
    assert len(cache_files) == 3  # distinct image path or params must not collide


def test_manifest_segmentation_dataset_uses_preprocess_cache_dir(tmp_path: Path):
    image = np.zeros((20, 20), dtype=np.uint8)
    image[5:15, 5:15] = 200
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[8:12, 8:12] = 255

    image_path = tmp_path / "cached_image.png"
    mask_path = tmp_path / "cached_mask.png"
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask).save(mask_path)

    manifest = pd.DataFrame(
        [
            {
                "patient_id": "p004",
                "phase": "diastole",
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "dataset": "synthetic",
                "subset": "train",
            }
        ]
    )
    cache_dir = tmp_path / "ds_cache"
    dataset = ManifestSegmentationDataset(
        manifest,
        image_size=(32, 32),
        preprocess_mode="wavelet",
        augment=False,
        preprocess_cache_dir=cache_dir,
    )
    sample_a = dataset[0]
    sample_b = dataset[0]
    assert torch.allclose(sample_a["image"], sample_b["image"])
    assert list((cache_dir / "wavelet").glob("*.npy"))


def test_manifest_segmentation_dataset_applies_mm_space_targets_per_patient_spacing(tmp_path: Path):
    image = np.zeros((40, 40), dtype=np.uint8)
    image[10:30, 10:30] = 200
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[15:25, 15:25] = 255

    image_path = tmp_path / "mm_image.png"
    mask_path = tmp_path / "mm_mask.png"
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask).save(mask_path)

    # Two "patients", same image, different native spacing -> the mm-space gaussian target
    # should resolve to a different pixel-space sigma for each, so outputs must differ.
    manifest = pd.DataFrame(
        [
            {
                "patient_id": "fine",
                "phase": "diastole",
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "dataset": "synthetic",
                "subset": "train",
                "spacing_row_mm": 1.0,
                "spacing_col_mm": 1.0,
            },
            {
                "patient_id": "coarse",
                "phase": "diastole",
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "dataset": "synthetic",
                "subset": "train",
                "spacing_row_mm": 2.0,
                "spacing_col_mm": 2.0,
            },
        ]
    )
    dataset = ManifestSegmentationDataset(
        manifest,
        image_size=(40, 40),
        preprocess_mode="gaussian",
        augment=False,
        mm_space_targets=MmSpaceFilterTargets(gaussian_sigma_mm=4.0),
    )
    fine_sample = dataset[0]
    coarse_sample = dataset[1]
    assert not torch.allclose(fine_sample["image"], coarse_sample["image"])
