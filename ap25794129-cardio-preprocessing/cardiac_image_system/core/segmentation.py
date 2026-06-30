from __future__ import annotations

import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, binary_opening, disk, remove_small_objects


def otsu_proxy_segmentation(image: np.ndarray, min_object_size: int = 64, morphology_radius: int = 2) -> np.ndarray:
    """Deterministic Otsu + morphology proxy segmentation.

    This is not a clinical segmentation method.
    """
    arr = np.asarray(image, dtype=np.float32)
    if arr.max() <= arr.min():
        return np.zeros_like(arr, dtype=bool)

    threshold = threshold_otsu(arr)
    mask = arr > threshold
    mask = remove_small_objects(mask, min_size=min_object_size)
    selem = disk(morphology_radius)
    mask = binary_closing(mask, selem)
    mask = binary_opening(mask, selem)
    return mask.astype(bool)
