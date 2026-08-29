from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Union

import nibabel as nib
import numpy as np
from PIL import Image
from skimage.io import imsave

PathLike = Union[str, Path]


@lru_cache(maxsize=16)
def _load_nifti_volume_cached(path_str: str) -> np.ndarray:
    """Decompress and load a full NIfTI volume as float32, cached by path.

    ``load_grayscale_image``/``load_label_image`` are called once per manifest row
    (i.e. once per 2D slice), but a 3D source volume (e.g. a full-resolution cardiac
    CT scan) is shared by every row drawn from the same patient. Without this cache,
    each row re-decompresses the entire .nii.gz volume just to keep one slice -- cheap
    for a handful of cine-MRI frames per ACDC patient, but a severe bottleneck for a
    single CT volume with dozens of sampled slices. Caching the decompressed array
    (not the slice) lets repeated accesses to the same patient reuse it; maxsize=16
    bounds memory to a handful of volumes at once (a full-resolution CT volume is on
    the order of several hundred MB) rather than growing unbounded across an epoch.
    """
    return nib.load(path_str).get_fdata(dtype=np.float32)


def _raw_slice_cache_key(path: str, slice_index: int | None, kind: str) -> str:
    payload = f"{kind}|{path}|{slice_index}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_slice_with_disk_cache(path, slice_index, cache_dir, kind, loader):
    """Disk-cache wrapper around a raw (unpreprocessed) NIfTI slice load.

    ``_load_nifti_volume_cached`` (an in-memory LRU) only helps when repeated accesses
    to the same patient land close together; a fully shuffled epoch over many patients
    defeats a small LRU every time, so training-time wall clock stayed dominated by
    repeated full-volume decompression even after adding it (measured: ~50 min/epoch
    unchanged on a 139-patient CT manifest). Unlike volume identity, a decoded 2D slice
    is tiny (one array, not a whole scan) and depends only on (path, slice_index), never
    on epoch, seed, or preprocessing mode/params -- so caching it to disk, keyed by
    those two fields alone, gives a guaranteed cache hit for every access after the
    first, regardless of shuffle order or how many epochs read it, without needing to
    hold every patient's volume in RAM at once (unlike a larger in-memory cache would).
    """
    cache_root = Path(cache_dir) / "_raw_slices" / kind
    key = _raw_slice_cache_key(str(path), slice_index, kind)
    cache_path = cache_root / f"{key}.npy"
    if cache_path.exists():
        return np.load(cache_path)
    arr = loader(path, slice_index=slice_index)
    cache_root.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_root / f"{key}.{os.getpid()}.tmp.npy"
    np.save(tmp_path, arr)
    os.replace(tmp_path, cache_path)  # atomic on POSIX and Windows NTFS
    return arr


def ensure_float01(image: np.ndarray) -> np.ndarray:
    """Convert image-like array to float32 and scale to [0, 1] if needed."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim > 2:
        arr = arr[..., 0]
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn:
        if 0.0 <= mx <= 1.0:
            return np.full_like(arr, fill_value=mx, dtype=np.float32)
        return np.full_like(arr, fill_value=1.0 if mx > 0.0 else 0.0, dtype=np.float32)
    if mn < 0.0 or mx > 1.0:
        arr = (arr - mn) / (mx - mn)
    return arr.astype(np.float32)


def _is_nifti_path(path: Path) -> bool:
    lower = path.name.lower()
    return lower.endswith(".nii") or lower.endswith(".nii.gz")


def _extract_nifti_slice(array: np.ndarray, slice_index: int | None = None) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        index = arr.shape[2] // 2 if slice_index is None else int(slice_index)
        if index < 0 or index >= arr.shape[2]:
            raise IndexError(f"slice_index {index} out of bounds for shape {arr.shape}")
        return arr[:, :, index]
    raise ValueError(f"Unsupported NIfTI dimensionality for 2D loading: {arr.shape}")


def get_nifti_in_plane_spacing(path: PathLike) -> tuple[float, float] | None:
    """Return (row_mm, col_mm) in-plane voxel spacing from a NIfTI header, or None if unavailable.

    Returns None for non-NIfTI inputs (PNG/NumPy sources carry no physical spacing metadata in
    this pipeline) rather than raising, since spacing is an optional enrichment, not a required
    field, for datasets or file formats that do not provide it.
    """
    path = Path(path)
    if not path.exists() or not _is_nifti_path(path):
        return None
    zooms = nib.load(path).header.get_zooms()
    if len(zooms) < 2:
        return None
    return (float(zooms[0]), float(zooms[1]))


def load_grayscale_image(path: PathLike, slice_index: int | None = None) -> np.ndarray:
    """Load image as float32 grayscale array in range [0, 1]."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    if path.suffix.lower() == ".npy":
        return ensure_float01(np.load(path))

    if _is_nifti_path(path):
        arr = _load_nifti_volume_cached(str(path))
        arr = _extract_nifti_slice(arr, slice_index=slice_index)
        return ensure_float01(arr)

    image = Image.open(path).convert("L")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return ensure_float01(arr)


def load_label_image(path: PathLike, slice_index: int | None = None) -> np.ndarray:
    """Load segmentation label image without intensity normalization."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Label image not found: {path}")

    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=np.int64)

    if _is_nifti_path(path):
        arr = _load_nifti_volume_cached(str(path))
        arr = _extract_nifti_slice(arr, slice_index=slice_index)
        return np.rint(arr).astype(np.int64)

    image = Image.open(path)
    arr = np.asarray(image)
    if arr.ndim > 2:
        arr = arr[..., 0]
    return arr.astype(np.int64)


def load_grayscale_image_cached(
    path: PathLike, slice_index: int | None, cache_dir: PathLike | None
) -> np.ndarray:
    """``load_grayscale_image``, disk-cached per (path, slice_index) when a NIfTI source
    and a ``cache_dir`` are given; identical to the uncached call otherwise (PNG/NumPy
    sources are already a single small file, so caching them again is not worthwhile)."""
    path = Path(path)
    if cache_dir is None or not _is_nifti_path(path):
        return load_grayscale_image(path, slice_index=slice_index)
    return _load_slice_with_disk_cache(path, slice_index, cache_dir, "image", load_grayscale_image)


def load_label_image_cached(
    path: PathLike, slice_index: int | None, cache_dir: PathLike | None
) -> np.ndarray:
    """``load_label_image``, disk-cached per (path, slice_index); see
    ``load_grayscale_image_cached``."""
    path = Path(path)
    if cache_dir is None or not _is_nifti_path(path):
        return load_label_image(path, slice_index=slice_index)
    return _load_slice_with_disk_cache(path, slice_index, cache_dir, "label", load_label_image)


def save_grayscale_image(path: PathLike, image: np.ndarray) -> None:
    """Save grayscale image after clipping to [0, 1]."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(image, 0.0, 1.0)
    imsave(path, (arr * 255).astype(np.uint8), check_contrast=False)
