from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from skimage.transform import resize
from torch.utils.data import Dataset

from cardiac_image_system.core.io import load_grayscale_image_cached, load_label_image_cached
from cardiac_image_system.core.preprocessing import (
    MmSpaceFilterTargets,
    PreprocessMode,
    PreprocessParams,
    preprocess_image,
    resolve_mm_space_params,
)


def _preprocess_cache_key(image_path: str, slice_index: int | None, mode: str, params: PreprocessParams) -> str:
    payload = f"{image_path}|{slice_index}|{mode}|{sorted(asdict(params).items())}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cached_preprocess_image(
    image: np.ndarray,
    cache_dir: Path,
    image_path: str,
    slice_index: int | None,
    mode: str,
    params: PreprocessParams,
) -> np.ndarray:
    """Preprocess-and-cache-to-disk wrapper around ``preprocess_image``.

    Preprocessing output depends only on (image_path, slice_index, mode, params), not on
    training seed, epoch, or split assignment, so it is identical across every seed and every
    epoch of a multiseed sweep for a fixed mode. CPU-bound modes (nlm, hybrid) are otherwise
    recomputed from scratch on every dataset access; caching the native-resolution preprocessed
    array to disk means each (image, mode, params) combination is computed once regardless of how
    many seeds or epochs subsequently read it. Resizing and augmentation are intentionally not
    cached: resizing is cheap, and augmentation is stochastic per-sample.
    """
    mode_dir = cache_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    key = _preprocess_cache_key(image_path, slice_index, mode, params)
    cache_path = mode_dir / f"{key}.npy"

    if cache_path.exists():
        return np.load(cache_path)

    processed = preprocess_image(image, mode=mode, params=params)
    tmp_path = mode_dir / f"{key}.{os.getpid()}.tmp.npy"
    np.save(tmp_path, processed)
    os.replace(tmp_path, cache_path)  # atomic on POSIX and Windows NTFS
    return processed


@dataclass(frozen=True)
class SegmentationSample:
    image: torch.Tensor
    mask: torch.Tensor
    patient_id: str
    phase: str
    dataset: str
    subset: str


def resize_pair(
    image: np.ndarray,
    mask: np.ndarray,
    target_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    image_resized = resize(
        image,
        target_size,
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)
    mask_resized = resize(
        mask.astype(np.float32),
        target_size,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )
    return image_resized, (mask_resized > 0.5).astype(np.float32)


def resize_label_map(mask: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    mask_resized = resize(
        mask.astype(np.float32),
        target_size,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )
    return np.rint(mask_resized).astype(np.int64)


LabelMode = Literal["binary", "multiclass"]


def apply_pair_augmentation(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
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


def apply_labelmap_augmentation(
    image: np.ndarray,
    label_map: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
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
    def __init__(
        self,
        manifest: pd.DataFrame,
        image_size: tuple[int, int] = (256, 256),
        preprocess_mode: PreprocessMode = "none",
        preprocess_params: PreprocessParams | None = None,
        augment: bool = False,
        seed: int = 42,
        label_mode: LabelMode = "binary",
        class_values: tuple[int, ...] | None = None,
        preprocess_cache_dir: str | Path | None = None,
        mm_space_targets: MmSpaceFilterTargets | None = None,
    ) -> None:
        if manifest.empty:
            raise ValueError("ManifestSegmentationDataset received an empty manifest")
        self.manifest = manifest.reset_index(drop=True).copy()
        self.image_size = tuple(int(x) for x in image_size)
        self.preprocess_mode = preprocess_mode
        self.preprocess_params = preprocess_params or PreprocessParams()
        self.augment = augment
        self.seed = int(seed)
        self.label_mode = label_mode
        self.class_values = tuple(int(x) for x in (class_values or (0, 1, 2, 3)))
        self.preprocess_cache_dir = Path(preprocess_cache_dir) if preprocess_cache_dir is not None else None
        self.mm_space_targets = mm_space_targets

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.manifest.iloc[index]
        slice_index = None
        if "slice_index" in row.index and not pd.isna(row["slice_index"]):
            slice_index = int(row["slice_index"])

        image = load_grayscale_image_cached(row["image_path"], slice_index, self.preprocess_cache_dir)
        raw_mask = load_label_image_cached(row["mask_path"], slice_index, self.preprocess_cache_dir)
        native_height, native_width = image.shape[:2]

        effective_params = self.preprocess_params
        if self.mm_space_targets is not None:
            native_spacing = None
            if "spacing_row_mm" in row.index and "spacing_col_mm" in row.index:
                spacing_row, spacing_col = row["spacing_row_mm"], row["spacing_col_mm"]
                if pd.notna(spacing_row) and pd.notna(spacing_col):
                    native_spacing = (float(spacing_row), float(spacing_col))
            effective_params = resolve_mm_space_params(self.preprocess_params, self.mm_space_targets, native_spacing)

        if self.preprocess_cache_dir is not None:
            image = cached_preprocess_image(
                image,
                cache_dir=self.preprocess_cache_dir,
                image_path=str(row["image_path"]),
                slice_index=slice_index,
                mode=self.preprocess_mode,
                params=effective_params,
            )
        else:
            image = preprocess_image(image, mode=self.preprocess_mode, params=effective_params)
        if self.label_mode == "binary":
            mask = (np.asarray(raw_mask, dtype=np.float32) > 0.0).astype(np.float32)
            image, mask = resize_pair(image, mask, target_size=self.image_size)
        else:
            image = resize(
                image,
                self.image_size,
                order=1,
                preserve_range=True,
                anti_aliasing=True,
            ).astype(np.float32)
            mask_labels = resize_label_map(raw_mask, target_size=self.image_size)
            invalid = ~np.isin(mask_labels, self.class_values)
            if invalid.any():
                mask_labels = mask_labels.copy()
                mask_labels[invalid] = 0
            mask = np.stack(
                [(mask_labels == class_value).astype(np.float32) for class_value in self.class_values],
                axis=0,
            )

        if self.augment:
            rng = np.random.default_rng(self.seed + index)
            if self.label_mode == "binary":
                image, mask = apply_pair_augmentation(image, mask, rng=rng)
            else:
                image, mask_labels = apply_labelmap_augmentation(image, mask_labels, rng=rng)
                mask = np.stack(
                    [(mask_labels == class_value).astype(np.float32) for class_value in self.class_values],
                    axis=0,
                )

        image_tensor = torch.from_numpy(np.ascontiguousarray(image[None, ...])).float()
        if self.label_mode == "binary":
            mask_tensor = torch.from_numpy(np.ascontiguousarray(mask[None, ...])).float()
            mask_labels_tensor = torch.from_numpy(np.ascontiguousarray((mask > 0.5).astype(np.int64)))
        else:
            mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).float()
            mask_labels_tensor = torch.from_numpy(np.ascontiguousarray(np.argmax(mask, axis=0).astype(np.int64)))

        # NaN (not None) sentinel for "spacing unavailable": default_collate cannot batch a
        # column of Python None mixed with floats, but happily batches NaN as a float tensor.
        resized_spacing_row_mm = float("nan")
        resized_spacing_col_mm = float("nan")
        if "spacing_row_mm" in row.index and "spacing_col_mm" in row.index:
            native_spacing_row = row["spacing_row_mm"]
            native_spacing_col = row["spacing_col_mm"]
            if pd.notna(native_spacing_row) and pd.notna(native_spacing_col):
                resized_spacing_row_mm = float(native_spacing_row) * (native_height / self.image_size[0])
                resized_spacing_col_mm = float(native_spacing_col) * (native_width / self.image_size[1])

        sample = {
            "image": image_tensor,
            "mask": mask_tensor,
            "mask_labels": mask_labels_tensor,
            "patient_id": str(row["patient_id"]),
            "phase": str(row["phase"]),
            "view": str(row["view"]) if "view" in row.index and pd.notna(row["view"]) else "unknown",
            "dataset": str(row["dataset"]) if "dataset" in row.index else "unknown",
            "subset": str(row["subset"]) if "subset" in row.index else "unknown",
            "image_path": str(Path(row["image_path"])),
            "mask_path": str(Path(row["mask_path"])),
            "resized_spacing_row_mm": resized_spacing_row_mm,
            "resized_spacing_col_mm": resized_spacing_col_mm,
        }
        if self.label_mode == "multiclass":
            sample["class_values"] = self.class_values
        return sample
