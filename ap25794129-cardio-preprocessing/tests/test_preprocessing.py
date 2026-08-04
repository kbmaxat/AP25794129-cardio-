import numpy as np

from cardiac_image_system.core.preprocessing import (
    MmSpaceFilterTargets,
    PreprocessParams,
    preprocess_image,
    resolve_mm_space_params,
)


def test_wavelet_level_is_configurable_and_changes_output():
    image = np.random.default_rng(0).random((64, 64)).astype("float32")
    out_level1 = preprocess_image(image, mode="wavelet", params=PreprocessParams(wavelet_level=1))
    out_level3 = preprocess_image(image, mode="wavelet", params=PreprocessParams(wavelet_level=3))
    assert out_level1.shape == out_level3.shape == image.shape
    assert not np.allclose(out_level1, out_level3)


def test_nlm_h_multiplier_is_configurable_and_changes_output():
    image = np.random.default_rng(0).random((64, 64)).astype("float32")
    out_light = preprocess_image(image, mode="nlm", params=PreprocessParams(nlm_h_multiplier=0.2))
    out_strong = preprocess_image(image, mode="nlm", params=PreprocessParams(nlm_h_multiplier=1.6))
    assert out_light.shape == out_strong.shape == image.shape
    assert not np.allclose(out_light, out_strong)


def test_clahe_clip_limit_is_configurable_and_changes_output():
    image = np.random.default_rng(0).random((64, 64)).astype("float32")
    out_low = preprocess_image(image, mode="clahe", params=PreprocessParams(clahe_clip_limit=0.01))
    out_high = preprocess_image(image, mode="clahe", params=PreprocessParams(clahe_clip_limit=0.06))
    assert out_low.shape == out_high.shape == image.shape
    assert not np.allclose(out_low, out_high)


def test_preprocess_modes_return_same_shape():
    image = np.random.default_rng(42).random((64, 64)).astype("float32")
    for mode in ["none", "gaussian", "wavelet", "nlm", "clahe", "hybrid"]:
        out = preprocess_image(image, mode=mode)
        assert out.shape == image.shape
        assert np.isfinite(out).all()
        assert out.min() >= 0.0
        assert out.max() <= 1.0


def test_constant_image_safe():
    image = np.ones((32, 32), dtype="float32")
    for mode in ["none", "wavelet", "hybrid"]:
        out = preprocess_image(image, mode=mode)
        assert out.shape == image.shape
        assert np.isfinite(out).all()


def test_resolve_mm_space_params_converts_using_mean_isotropic_spacing():
    base = PreprocessParams(gaussian_sigma=1.0, nlm_patch_size=5, nlm_patch_distance=6, clahe_kernel_size=16)
    targets = MmSpaceFilterTargets(gaussian_sigma_mm=2.0, nlm_patch_size_mm=5.0)
    # mean spacing = (1.0 + 3.0) / 2 = 2.0 mm/pixel
    resolved = resolve_mm_space_params(base, targets, spacing_mm=(1.0, 3.0))
    assert resolved.gaussian_sigma == 1.0  # 2.0mm / 2.0mm-per-px = 1.0px
    assert resolved.nlm_patch_size == round(5.0 / 2.0)
    # unset targets fall back to the base pixel-space value unchanged
    assert resolved.nlm_patch_distance == base.nlm_patch_distance
    assert resolved.clahe_kernel_size == base.clahe_kernel_size


def test_resolve_mm_space_params_returns_base_unchanged_without_spacing():
    base = PreprocessParams(gaussian_sigma=1.0)
    targets = MmSpaceFilterTargets(gaussian_sigma_mm=2.0)
    resolved = resolve_mm_space_params(base, targets, spacing_mm=None)
    assert resolved is base


def test_resolve_mm_space_params_holds_physical_strength_constant_across_patients():
    base = PreprocessParams(gaussian_sigma=1.0)
    targets = MmSpaceFilterTargets(gaussian_sigma_mm=3.0)
    # two "patients" with different native pixel spacing should get different pixel-space sigma
    # values that represent the *same* 3.0mm physical blur radius
    fine_spacing_patient = resolve_mm_space_params(base, targets, spacing_mm=(1.0, 1.0))
    coarse_spacing_patient = resolve_mm_space_params(base, targets, spacing_mm=(2.0, 2.0))
    assert fine_spacing_patient.gaussian_sigma == 3.0
    assert coarse_spacing_patient.gaussian_sigma == 1.5
    assert fine_spacing_patient.gaussian_sigma != coarse_spacing_patient.gaussian_sigma
