from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from cardiac_image_system.core.torch_data import ManifestSegmentationDataset


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
