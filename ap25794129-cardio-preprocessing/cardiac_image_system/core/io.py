from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from skimage.io import imsave

PathLike = Union[str, Path]


def ensure_float01(image: np.ndarray) -> np.ndarray:
    """Convert image-like array to float32 and scale to [0, 1] if needed."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim > 2:
        arr = arr[..., 0]
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn:
        return np.zeros_like(arr, dtype=np.float32)
    if mn < 0.0 or mx > 1.0:
        arr = (arr - mn) / (mx - mn)
    return arr.astype(np.float32)


def load_grayscale_image(path: PathLike) -> np.ndarray:
    """Load image as float32 grayscale array in range [0, 1]."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    if path.suffix.lower() == ".npy":
        return ensure_float01(np.load(path))

    image = Image.open(path).convert("L")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return ensure_float01(arr)


def save_grayscale_image(path: PathLike, image: np.ndarray) -> None:
    """Save grayscale image after clipping to [0, 1]."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(image, 0.0, 1.0)
    imsave(path, (arr * 255).astype(np.uint8), check_contrast=False)
