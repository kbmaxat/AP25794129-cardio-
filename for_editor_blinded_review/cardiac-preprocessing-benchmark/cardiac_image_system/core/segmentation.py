from __future__ import annotations

import numpy as np
from scipy.ndimage import label
from skimage.filters import threshold_otsu
from skimage.morphology import closing, disk, opening


def _remove_small_components(mask: np.ndarray, min_object_size: int) -> np.ndarray:
    if min_object_size <= 1:
        return mask.astype(bool)
    labeled, num_labels = label(mask.astype(bool))
    if num_labels == 0:
        return mask.astype(bool)
    counts = np.bincount(labeled.ravel())
    keep = np.zeros_like(counts, dtype=bool)
    keep[0] = False
    keep[counts >= min_object_size] = True
    return keep[labeled]


def otsu_proxy_segmentation(image: np.ndarray, min_object_size: int = 64, morphology_radius: int = 2) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.max() <= arr.min():
        return np.zeros_like(arr, dtype=bool)
    threshold = threshold_otsu(arr)
    mask = arr > threshold
    mask = _remove_small_components(mask, min_object_size=min_object_size)
    selem = disk(morphology_radius)
    mask = closing(mask, selem)
    mask = opening(mask, selem)
    return mask.astype(bool)
