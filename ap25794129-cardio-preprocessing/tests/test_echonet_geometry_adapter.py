import numpy as np
import pytest

from cardiac_image_system.research_temporal.echonet_geometry import (
    CamusEchoNetAxisAdapterV1,
)


def test_adapter_transposes_input_and_restores_prediction_exactly():
    adapter = CamusEchoNetAxisAdapterV1()
    image = np.arange(3 * 5, dtype=np.float32).reshape(3, 5)
    probability_map = np.arange(5 * 3, dtype=np.float32).reshape(5, 3) / 15

    adapted = adapter.prepare_input(image)
    restored = adapter.restore_prediction(probability_map)

    assert adapter.version == "camus-echonet-axis-transpose-v1"
    assert adapted.shape == (5, 3)
    assert restored.shape == image.shape
    np.testing.assert_array_equal(adapted, image.T)
    np.testing.assert_array_equal(restored, probability_map.T)
    assert adapted.flags.c_contiguous
    assert restored.flags.c_contiguous
    assert not np.shares_memory(adapted, image)
    assert not np.shares_memory(restored, probability_map)


def test_adapter_round_trip_preserves_asymmetric_landmark_coordinates():
    adapter = CamusEchoNetAxisAdapterV1()
    image = np.zeros((7, 11), dtype=np.uint8)
    image[2, 8] = 1

    restored = adapter.restore_prediction(adapter.prepare_input(image))

    np.testing.assert_array_equal(restored, image)
    assert np.argwhere(restored == 1).tolist() == [[2, 8]]


@pytest.mark.parametrize(
    "value",
    [
        np.zeros((0, 3), dtype=np.float32),
        np.zeros((3,), dtype=np.float32),
        np.zeros((1, 3, 5), dtype=np.float32),
    ],
)
def test_adapter_rejects_empty_or_non_2d_arrays(value):
    adapter = CamusEchoNetAxisAdapterV1()
    with pytest.raises(ValueError):
        adapter.prepare_input(value)
    with pytest.raises(ValueError):
        adapter.restore_prediction(value)
