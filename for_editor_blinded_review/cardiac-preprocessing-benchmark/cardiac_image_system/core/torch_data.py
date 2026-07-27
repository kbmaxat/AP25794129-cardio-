from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from skimage.transform import resize
from torch.utils.data import Dataset

from cardiac_image_system.core.io import load_grayscale_image, load_label_image
from cardiac_image_system.core.preprocessing import PreprocessMode, preprocess_image


@dataclass(frozen=True)
class SegmentationSample:
    image: torch.Tensor
    mask: torch.Tensor
    patient_id: str
    phase: str
    dataset: str
    subset: str


def resize_pair(image: np.ndarray, mask: np.ndarray, target_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    image_resized = resize(image, target_size, order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
    mask_resized = resize(mask.astype(np.float32), target_size, order=0, preserve_range=True, anti_aliasing=False)
    return image_resized, (mask_resized > 0.5).astype(np.float32)


def resize_label_map(mask: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    mask_resized = resize(mask.astype(np.float32), target_size, order=0, preserve_range=True, anti_aliasing=False)
    return np.rint(mask_resized).astype(np.int64)


LabelMode = Literal["binary", "multiclass"]


def apply_pair_augmentation(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    if rng.random() < 0.5:
        image = np.flipud(image)
        mask = np.flipud(mask)

    k = int(rng.integers(0, 4))
    if k:
        image = np.rot90(image, k)
        mask = np.rot90(mask, k)

    if rng.random() < 0.3:
        gain = float(rng.uniform(0.9, 1.1))
        bias = float(rng.uniform(-0.05, 0.05))
        image = np.clip(image * gain + bias, 0.0, 1.0)

    if rng.random() < 0.2:
        noise = rng.normal(loc=0.0, scale=0.015, size=image.shape).astype(np.float32)
        image = np.clip(image + noise, 0.0, 1.0)

    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


def apply_labelmap_augmentation(image: np.ndarray, label_map: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        image = np.fliplr(image)
        label_map = np.fliplr(label_map)
    if rng.random() < 0.5:
        image = np.flipud(image)
        label_map = np.flipud(label_map)

    k = int(rng.integers(0, 4))
    if k:
        image = np.rot90(image, k)
        label_map = np.rot90(label_map, k)

    if rng.random() < 0.3:
        gain = float(rng.uniform(0.9, 1.1))
        bias = float(rng.uniform(-0.05, 0.05))
        image = np.clip(image * gain + bias, 0.0, 1.0)

    if rng.random() < 0.2:
        noise = rng.normal(loc=0.0, scale=0.015, size=image.shape).astype(np.float32)
        image = np.clip(image + noise, 0.0, 1.0)

    return np.ascontiguousarray(image), np.ascontiguousarray(label_map)


class ManifestSegmentationDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, image_size: tuple[int, int] = (256, 256), preprocess_mode: PreprocessMode = "none", augment: bool = False, seed: int = 42, label_mode: LabelMode = "binary", class_values: tuple[int, ...] | None = None) -> None:
        if manifest.empty:
            raise ValueError("ManifestSegmentationDataset received an empty manifest")
        self.manifest = manifest.reset_index(drop=True).copy()
        self.image_size = tuple(int(x) for x in image_size)
        self.preprocess_mode = preprocess_mode
        self.augment = augment
        self.seed = int(seed)
        self.label_mode = label_mode
        self.class_values = tuple(int(x) for x in (class_values or (0, 1, 2, 3)))

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.manifest.iloc[index]
        slice_index = None
        if "slice_index" in row.index and not pd.isna(row["slice_index"]):
            slice_index = int(row["slice_index"])

        image = load_grayscale_image(row["image_path"], slice_index=slice_index)
        raw_mask = load_label_image(row["mask_path"], slice_index=slice_index)

        image = preprocess_image(image, mode=self.preprocess_mode)
        if self.label_mode == "binary":
            mask = (np.asarray(raw_mask, dtype=np.float32) > 0.0).astype(np.float32)
            image, mask = resize_pair(image, mask, target_size=self.image_size)
        else:
            image = resize(image, self.image_size, order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
            mask_labels = resize_label_map(raw_mask, target_size=self.image_size)
            invalid = ~np.isin(mask_labels, self.class_values)
            if invalid.any():
                mask_labels = mask_labels.copy()
                mask_labels[invalid] = 0
            mask = np.stack([(mask_labels == class_value).astype(np.float32) for class_value in self.class_values], axis=0)

        if self.augment:
            rng = np.random.default_rng(self.seed + index)
            if self.label_mode == "binary":
                image, mask = apply_pair_augmentation(image, mask, rng=rng)
            else:
                image, mask_labels = apply_labelmap_augmentation(image, mask_labels, rng=rng)
                mask = np.stack([(mask_labels == class_value).astype(np.float32) for class_value in self.class_values], axis=0)

        image_tensor = torch.from_numpy(np.ascontiguousarray(image[None, ...])).float()
        if self.label_mode == "binary":
            mask_tensor = torch.from_numpy(np.ascontiguousarray(mask[None, ...])).float()
            mask_labels_tensor = torch.from_numpy(np.ascontiguousarray((mask > 0.5).astype(np.int64)))
        else:
            mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).float()
            mask_labels_tensor = torch.from_numpy(np.ascontiguousarray(np.argmax(mask, axis=0).astype(np.int64)))

        sample = {
            "image": image_tensor,
            "mask": mask_tensor,
            "mask_labels": mask_labels_tensor,
            "patient_id": str(row["patient_id"]),
            "phase": str(row["phase"]),
            "dataset": str(row["dataset"]) if "dataset" in row.index else "unknown",
            "subset": str(row["subset"]) if "subset" in row.index else "unknown",
            "image_path": str(Path(row["image_path"])),
            "mask_path": str(Path(row["mask_path"])),
        }
        if self.label_mode == "multiclass":
            sample["class_values"] = self.class_values
        return sample
