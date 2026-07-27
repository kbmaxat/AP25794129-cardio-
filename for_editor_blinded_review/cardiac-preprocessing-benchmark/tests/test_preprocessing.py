import numpy as np

from cardiac_image_system.core.preprocessing import preprocess_image


def test_preprocess_modes_return_same_shape():
    image = np.random.default_rng(42).random((64, 64)).astype("float32")
    for mode in ["none", "gaussian", "wavelet", "nlm", "clahe", "hybrid"]:
        out = preprocess_image(image, mode=mode)
        assert out.shape == image.shape
        assert np.isfinite(out).all()
