from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pywt
from scipy.ndimage import gaussian_filter as scipy_gaussian_filter
from skimage import exposure
from skimage.restoration import denoise_nl_means, estimate_sigma

from cardiac_image_system.core.io import ensure_float01

PreprocessMode = Literal["none", "gaussian", "wavelet", "nlm", "clahe", "hybrid"]


@dataclass(frozen=True)
class PreprocessParams:
    gaussian_sigma: float = 1.0
    wavelet_name: str = "haar"
    wavelet_level: int = 2
    nlm_h: float | None = None
    nlm_h_multiplier: float = 0.8
    nlm_patch_size: int = 5
    nlm_patch_distance: int = 6
    clahe_clip_limit: float = 0.03
    clahe_kernel_size: int = 16


@dataclass(frozen=True)
class MmSpaceFilterTargets:
    """Physical-scale (millimeter) targets for the pixel-space filter parameters in
    ``PreprocessParams`` that represent a spatial extent (kernel/patch/window size).

    The primary benchmark parameterizes these filters in pixels, applied at each image's native
    resolution; because ACDC and CAMUS (and different patients within ACDC) differ in native pixel
    spacing, a fixed pixel-space parameter corresponds to a different physical filter strength for
    different patients. Setting an mm target here and converting it per-image via
    ``resolve_mm_space_params`` holds the physical filter strength constant across patients
    instead. Fields left as ``None`` fall back to the corresponding pixel-space default in the
    base ``PreprocessParams`` unchanged (not converted).
    """

    gaussian_sigma_mm: float | None = None
    nlm_patch_size_mm: float | None = None
    nlm_patch_distance_mm: float | None = None
    clahe_kernel_size_mm: float | None = None


def resolve_mm_space_params(
    base_params: PreprocessParams,
    targets: MmSpaceFilterTargets,
    spacing_mm: tuple[float, float] | None,
) -> PreprocessParams:
    """Return a ``PreprocessParams`` with mm-space ``targets`` converted to pixels for this image.

    Anisotropic in-plane spacing (``spacing_mm`` may have unequal row/column components) is
    reduced to a single isotropic mm-per-pixel scale via the mean of the two axes, since the
    filters this project implements (Gaussian sigma, NLM patch/search windows, CLAHE tile size)
    are themselves isotropic pixel-space parameters, not directionally aware. When ``spacing_mm``
    is ``None`` (spacing unavailable for this image), ``base_params`` is returned unchanged --
    mm-space conversion silently does nothing rather than raising, since spacing is not available
    for every source in this pipeline.
    """
    if spacing_mm is None:
        return base_params

    mm_per_pixel = (spacing_mm[0] + spacing_mm[1]) / 2.0
    if mm_per_pixel <= 0:
        return base_params

    updates: dict[str, float | int] = {}
    if targets.gaussian_sigma_mm is not None:
        updates["gaussian_sigma"] = targets.gaussian_sigma_mm / mm_per_pixel
    if targets.nlm_patch_size_mm is not None:
        updates["nlm_patch_size"] = max(1, round(targets.nlm_patch_size_mm / mm_per_pixel))
    if targets.nlm_patch_distance_mm is not None:
        updates["nlm_patch_distance"] = max(1, round(targets.nlm_patch_distance_mm / mm_per_pixel))
    if targets.clahe_kernel_size_mm is not None:
        updates["clahe_kernel_size"] = max(1, round(targets.clahe_kernel_size_mm / mm_per_pixel))

    if not updates:
        return base_params
    return replace(base_params, **updates)


def normalize_minmax(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx - mn < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn + eps)).astype(np.float32)


def apply_gaussian(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    image = ensure_float01(image)
    return ensure_float01(scipy_gaussian_filter(image, sigma=sigma))


def _soft_threshold(coeff: np.ndarray, threshold: float) -> np.ndarray:
    if not np.isfinite(threshold) or threshold <= 0.0:
        return np.asarray(coeff, dtype=np.float32)
    coeff = np.asarray(coeff, dtype=np.float32)
    magnitude = np.abs(coeff)
    shrink = np.maximum(magnitude - threshold, 0.0)
    return np.sign(coeff) * shrink


def apply_wavelet_denoise(image: np.ndarray, wavelet_name: str = "haar", level: int = 2) -> np.ndarray:
    image = ensure_float01(image)
    coeffs = pywt.wavedec2(image, wavelet=wavelet_name, level=level)
    detail_coeffs = coeffs[-1]
    detail_arr = np.concatenate([c.ravel() for c in detail_coeffs])
    sigma_hat = np.median(np.abs(detail_arr)) / 0.6745 if detail_arr.size else 0.0
    threshold = sigma_hat * np.sqrt(2.0 * np.log(image.size + 1))

    new_coeffs = [coeffs[0]]
    for detail in coeffs[1:]:
        new_coeffs.append(tuple(_soft_threshold(c, threshold) for c in detail))

    reconstructed = pywt.waverec2(new_coeffs, wavelet=wavelet_name)
    reconstructed = reconstructed[: image.shape[0], : image.shape[1]]
    return ensure_float01(reconstructed)


def apply_nlm(
    image: np.ndarray,
    h: float | None = None,
    patch_size: int = 5,
    patch_distance: int = 6,
    h_sigma_multiplier: float = 0.8,
) -> np.ndarray:
    image = ensure_float01(image)
    if float(np.std(image)) <= 1e-8:
        return image.astype(np.float32, copy=True)
    sigma = float(np.mean(estimate_sigma(image, channel_axis=None)))
    h_value = h if h is not None else max(h_sigma_multiplier * sigma, 0.01)
    out = denoise_nl_means(
        image,
        h=h_value,
        patch_size=patch_size,
        patch_distance=patch_distance,
        fast_mode=True,
        channel_axis=None,
        preserve_range=True,
    )
    return ensure_float01(out)


def apply_clahe(image: np.ndarray, clip_limit: float = 0.03, kernel_size: int = 16) -> np.ndarray:
    image = ensure_float01(image)
    return ensure_float01(exposure.equalize_adapthist(image, clip_limit=clip_limit, kernel_size=kernel_size))


def preprocess_image(image: np.ndarray, mode: PreprocessMode, params: PreprocessParams | None = None) -> np.ndarray:
    params = params or PreprocessParams()
    x = normalize_minmax(image)

    if mode == "none":
        return x
    if mode == "gaussian":
        return apply_gaussian(x, sigma=params.gaussian_sigma)
    if mode == "wavelet":
        return apply_wavelet_denoise(x, params.wavelet_name, params.wavelet_level)
    if mode == "nlm":
        return apply_nlm(x, params.nlm_h, params.nlm_patch_size, params.nlm_patch_distance, params.nlm_h_multiplier)
    if mode == "clahe":
        return apply_clahe(x, params.clahe_clip_limit, params.clahe_kernel_size)
    if mode == "hybrid":
        x = apply_wavelet_denoise(x, params.wavelet_name, params.wavelet_level)
        x = apply_nlm(x, params.nlm_h, params.nlm_patch_size, params.nlm_patch_distance, params.nlm_h_multiplier)
        x = apply_clahe(x, params.clahe_clip_limit, params.clahe_kernel_size)
        return x
    raise ValueError(f"Unknown preprocessing mode: {mode}")
