"""Deterministic paired augmentation for short-axis cardiac MRI volumes."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class AugmentationParameters:
    rotation_degrees: float
    translation_y: float
    translation_x: float
    gamma: float
    intensity_scale: float
    intensity_shift: float
    bias_y: float
    bias_x: float
    rician_sigma: float


def sample_parameters(rng: np.random.Generator) -> AugmentationParameters:
    return AugmentationParameters(
        rotation_degrees=float(rng.uniform(-10.0, 10.0)),
        translation_y=float(rng.uniform(-5.0, 5.0)),
        translation_x=float(rng.uniform(-5.0, 5.0)),
        gamma=float(rng.uniform(0.85, 1.15)),
        intensity_scale=float(rng.uniform(0.90, 1.10)),
        intensity_shift=float(rng.uniform(-0.04, 0.04)),
        bias_y=float(rng.uniform(-0.10, 0.10)),
        bias_x=float(rng.uniform(-0.10, 0.10)),
        rician_sigma=float(rng.uniform(0.0, 0.025)),
    )


def augment_pair(
    image: np.ndarray,
    label: np.ndarray,
    parameters: AugmentationParameters,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one spatial transform to image/label and intensity transforms to image."""
    if image.shape != label.shape or image.ndim != 3:
        raise ValueError("Expected paired 3D arrays with equal shapes")

    image_out = ndimage.rotate(
        image,
        parameters.rotation_degrees,
        axes=(1, 2),
        reshape=False,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    label_out = ndimage.rotate(
        label,
        parameters.rotation_degrees,
        axes=(1, 2),
        reshape=False,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    )

    shift = (0.0, parameters.translation_y, parameters.translation_x)
    image_out = ndimage.shift(
        image_out,
        shift,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    label_out = ndimage.shift(
        label_out,
        shift,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    ).astype(np.uint8)

    image_out = np.clip(image_out, 0.0, 1.0) ** parameters.gamma
    y = np.linspace(-1.0, 1.0, image.shape[1], dtype=np.float32)
    x = np.linspace(-1.0, 1.0, image.shape[2], dtype=np.float32)
    bias = (
        1.0
        + parameters.bias_y * y[None, :, None]
        + parameters.bias_x * x[None, None, :]
    )
    image_out = (
        image_out * bias * parameters.intensity_scale + parameters.intensity_shift
    )

    if parameters.rician_sigma > 0:
        noise_real = rng.normal(0.0, parameters.rician_sigma, image.shape)
        noise_imag = rng.normal(0.0, parameters.rician_sigma, image.shape)
        image_out = np.sqrt((image_out + noise_real) ** 2 + noise_imag**2)

    return np.clip(image_out, 0.0, 1.0).astype(np.float32), label_out


def parameters_as_dict(parameters: AugmentationParameters) -> dict[str, float]:
    return asdict(parameters)


def validate_pair(
    image: np.ndarray,
    label: np.ndarray,
    reference_label: np.ndarray | None = None,
) -> dict[str, float | int | list[int]]:
    if image.shape != label.shape:
        raise ValueError("Image and label shapes differ")
    if not np.isfinite(image).all():
        raise ValueError("Image contains NaN or infinity")
    unique_labels = np.unique(label)
    if not set(unique_labels.tolist()).issubset({0, 1, 2}):
        raise ValueError(f"Unexpected labels: {unique_labels.tolist()}")
    if not np.any(label == 1) or not np.any(label == 2):
        raise ValueError("Cavity or myocardium disappeared after augmentation")
    if image.min() < 0.0 or image.max() > 1.0:
        raise ValueError("Image values are outside [0, 1]")

    result: dict[str, float | int | list[int]] = {
        "shape": list(image.shape),
        "cavity_voxels": int(np.sum(label == 1)),
        "myocardium_voxels": int(np.sum(label == 2)),
        "image_min": float(image.min()),
        "image_max": float(image.max()),
        "image_mean": float(image.mean()),
    }
    if reference_label is not None:
        for class_id, name in ((1, "cavity"), (2, "myocardium")):
            original = int(np.sum(reference_label == class_id))
            current = int(np.sum(label == class_id))
            ratio = current / original
            if not 0.80 <= ratio <= 1.20:
                raise ValueError(
                    f"{name} voxel ratio {ratio:.3f} is outside QC limits"
                )
            result[f"{name}_volume_ratio"] = float(ratio)
    return result
