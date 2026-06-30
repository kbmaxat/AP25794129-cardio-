from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(reference: np.ndarray, image: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(reference, image, data_range=1.0))


def ssim(reference: np.ndarray, image: np.ndarray) -> float:
    return float(structural_similarity(reference, image, data_range=1.0))


def dice(mask_true: np.ndarray, mask_pred: np.ndarray, eps: float = 1e-8) -> float:
    true = np.asarray(mask_true).astype(bool)
    pred = np.asarray(mask_pred).astype(bool)
    inter = np.logical_and(true, pred).sum()
    denom = true.sum() + pred.sum()
    return float((2.0 * inter + eps) / (denom + eps))


def iou(mask_true: np.ndarray, mask_pred: np.ndarray, eps: float = 1e-8) -> float:
    true = np.asarray(mask_true).astype(bool)
    pred = np.asarray(mask_pred).astype(bool)
    inter = np.logical_and(true, pred).sum()
    union = np.logical_or(true, pred).sum()
    return float((inter + eps) / (union + eps))


def relative_area_error(mask_true: np.ndarray, mask_pred: np.ndarray, eps: float = 1e-8) -> float:
    true_area = np.asarray(mask_true).astype(bool).sum()
    pred_area = np.asarray(mask_pred).astype(bool).sum()
    return float(abs(pred_area - true_area) / (true_area + eps))


def _surface(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    eroded = binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
    return mask ^ eroded


def hd95(mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
    """Compute 95th percentile Hausdorff distance for binary masks."""
    true = np.asarray(mask_true).astype(bool)
    pred = np.asarray(mask_pred).astype(bool)

    if true.sum() == 0 and pred.sum() == 0:
        return 0.0
    if true.sum() == 0 or pred.sum() == 0:
        return float("inf")

    true_surface = _surface(true)
    pred_surface = _surface(pred)

    dt_true = distance_transform_edt(~true_surface)
    dt_pred = distance_transform_edt(~pred_surface)

    distances_true_to_pred = dt_pred[true_surface]
    distances_pred_to_true = dt_true[pred_surface]
    distances = np.concatenate([distances_true_to_pred, distances_pred_to_true])

    if distances.size == 0:
        return 0.0
    return float(np.percentile(distances, 95))
