import numpy as np

from cardiac_image_system.core.metrics import dice, iou, relative_area_error


def test_dice_perfect():
    mask = np.array([[1, 0], [0, 1]], dtype=bool)
    assert dice(mask, mask) == 1.0


def test_iou_perfect():
    mask = np.array([[1, 0], [0, 1]], dtype=bool)
    assert iou(mask, mask) == 1.0


def test_relative_area_error_zero_for_same_mask():
    mask = np.array([[1, 0], [0, 1]], dtype=bool)
    assert relative_area_error(mask, mask) == 0.0
