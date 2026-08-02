import numpy as np

from cardiac_image_system.core.preprocessing import PreprocessParams, preprocess_image


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
