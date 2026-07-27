import numpy as np

from cardiac_image_system.core.metrics import dice, iou, psnr


def test_dice_perfect():
    mask = np.array([[1, 0], [0, 1]], dtype=bool)
    assert dice(mask, mask) == 1.0


def test_iou_perfect():
    mask = np.array([[1, 0], [0, 1]], dtype=bool)
    assert iou(mask, mask) == 1.0


def test_psnr_identical_images_is_infinite():
    image = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    assert psnr(image, image) == float("inf")
