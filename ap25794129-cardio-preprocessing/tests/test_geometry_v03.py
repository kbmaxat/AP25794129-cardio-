import numpy as np
import pytest

from cardiac_image_system.research_temporal.geometry import (
    apply_letterbox,
    compute_letterbox_transform,
    map_points_from_letterbox,
    map_points_to_letterbox,
)


@pytest.mark.parametrize(
    "native_shape",
    [
        (549, 389),  # measured CAMUS patient0001, aspect ratio ~1.41
        (388, 389),  # measured CAMUS patient0015, near-square
        (787, 649),  # measured CAMUS patient0008, aspect ratio ~1.21
        (400, 400),  # exactly square, transform must be a no-op besides identity scale
        (600, 128),  # extreme aspect ratio, not present in CAMUS but stresses the math
    ],
)
def test_point_round_trip_is_exact_up_to_floating_point(native_shape):
    size = 256
    transform = compute_letterbox_transform(native_shape, size)
    rng = np.random.default_rng(0)
    h, w = native_shape
    points = np.stack([rng.uniform(0, w, size=50), rng.uniform(0, h, size=50)], axis=-1)

    forward = map_points_to_letterbox(points, transform)
    # every mapped point must land inside the padded square
    assert (forward[..., 0] >= -1e-6).all() and (forward[..., 0] <= size + 1e-6).all()
    assert (forward[..., 1] >= -1e-6).all() and (forward[..., 1] <= size + 1e-6).all()

    back = map_points_from_letterbox(forward, transform)
    np.testing.assert_allclose(back, points, atol=1e-6)


def test_square_input_has_no_padding():
    transform = compute_letterbox_transform((256, 256), 256)
    assert transform.pad_top == 0
    assert transform.pad_left == 0
    assert transform.scale == pytest.approx(1.0)


def test_letterbox_preserves_aspect_ratio_unlike_square_stretch():
    # A synthetic mask with a single foreground pixel placed off-center in a rectangular image.
    native_shape = (200, 100)  # height, width; aspect ratio 2.0
    size = 128
    mask = np.zeros(native_shape, dtype=np.float32)
    mask[150, 20] = 1.0  # a known landmark, well away from the center

    transform = compute_letterbox_transform(native_shape, size)
    resized_mask = apply_letterbox(mask, transform, interpolation=0)  # cv2.INTER_NEAREST == 0

    # recover the landmark location in the padded square via forward point mapping
    expected_xy = map_points_to_letterbox(np.array([[20, 150]]), transform)[0]
    ys, xs = np.nonzero(resized_mask)
    assert len(ys) >= 1
    recovered_xy = np.array([xs.mean(), ys.mean()])
    np.testing.assert_allclose(recovered_xy, expected_xy, atol=1.5)

    # inverse mapping must bring it back close to the native landmark
    native_back = map_points_from_letterbox(expected_xy, transform)
    np.testing.assert_allclose(native_back, [20, 150], atol=1e-6)


def test_invalid_shapes_are_rejected():
    with pytest.raises(ValueError):
        compute_letterbox_transform((0, 10), 64)
    with pytest.raises(ValueError):
        compute_letterbox_transform((10, 10), 0)
