"""Aspect-ratio preserving resize for the geometry-preservation hypothesis (protocol v0.3 candidate).

The current pipeline (`research_temporal.data.load_cases`) resizes every image and mask directly to a
fixed square with `cv2.resize(..., (size, size))`. For a non-square source this is an anisotropic
stretch: height and width are scaled by different factors. This module implements the alternative
control described in `research/temporal_v01/protocol_v03_candidate.md`: resize with a single shared
scale factor (the longer side maps to `size`), then pad the shorter side to reach a square. The forward
transform and its exact inverse are provided together so that predictions produced on the padded square
can be mapped back into the native pixel coordinate system before comparison with the original masks.

No training or dataset access happens in this module; it only implements and documents the geometric
transform so it can be unit-tested in isolation first.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxTransform:
    """Parameters of a single aspect-ratio-preserving resize-and-pad operation.

    Attributes:
        scale: shared scale factor applied to both axes of the native image.
        pad_top, pad_left: offset in output pixels of the resized content within the padded square.
        native_shape: (height, width) of the source image before this transform.
        size: side length of the padded square output.
    """

    scale: float
    pad_top: int
    pad_left: int
    native_shape: tuple[int, int]
    size: int


def compute_letterbox_transform(native_shape: tuple[int, int], size: int) -> LetterboxTransform:
    h, w = native_shape
    if h <= 0 or w <= 0:
        raise ValueError("native_shape must be positive")
    if size <= 0:
        raise ValueError("size must be positive")
    scale = size / max(h, w)
    resized_h = int(round(h * scale))
    resized_w = int(round(w * scale))
    pad_top = (size - resized_h) // 2
    pad_left = (size - resized_w) // 2
    return LetterboxTransform(scale=scale, pad_top=pad_top, pad_left=pad_left, native_shape=(h, w), size=size)


def apply_letterbox(image: np.ndarray, transform: LetterboxTransform, interpolation: int) -> np.ndarray:
    """Resize `image` with a single shared scale factor and pad it to `transform.size` x `transform.size`.

    `interpolation` should be `cv2.INTER_AREA` for continuous images and `cv2.INTER_NEAREST` for
    label masks, mirroring the choices already used for the control (square-stretch) pipeline.
    """
    h, w = transform.native_shape
    if image.shape[:2] != (h, w):
        raise ValueError("image shape does not match the shape used to compute the transform")
    resized_h = int(round(h * transform.scale))
    resized_w = int(round(w * transform.scale))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=interpolation)
    out_shape = (transform.size, transform.size) + image.shape[2:]
    padded = np.zeros(out_shape, dtype=image.dtype)
    padded[
        transform.pad_top: transform.pad_top + resized_h,
        transform.pad_left: transform.pad_left + resized_w,
        ...,
    ] = resized
    return padded


def map_points_to_letterbox(points_xy: np.ndarray, transform: LetterboxTransform) -> np.ndarray:
    """Map native (x, y) pixel coordinates forward into the padded square coordinate system."""
    points_xy = np.asarray(points_xy, dtype=np.float64)
    out = points_xy * transform.scale
    out[..., 0] += transform.pad_left
    out[..., 1] += transform.pad_top
    return out


def map_points_from_letterbox(points_xy: np.ndarray, transform: LetterboxTransform) -> np.ndarray:
    """Exact inverse of `map_points_to_letterbox`: map padded-square coordinates back to native pixels."""
    points_xy = np.asarray(points_xy, dtype=np.float64)
    out = points_xy.copy()
    out[..., 0] -= transform.pad_left
    out[..., 1] -= transform.pad_top
    out /= transform.scale
    return out
