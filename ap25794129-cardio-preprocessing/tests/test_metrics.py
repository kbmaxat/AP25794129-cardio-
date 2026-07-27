import numpy as np

from cardiac_image_system.core.metrics import dice, iou, multiclass_overlap_metrics, psnr, relative_area_error


def test_dice_perfect():
    mask = np.array([[1, 0], [0, 1]], dtype=bool)
    assert dice(mask, mask) == 1.0


def test_iou_perfect():
    mask = np.array([[1, 0], [0, 1]], dtype=bool)
    assert iou(mask, mask) == 1.0


def test_relative_area_error_zero_for_same_mask():
    mask = np.array([[1, 0], [0, 1]], dtype=bool)
    assert relative_area_error(mask, mask) == 0.0


def test_psnr_identical_images_is_infinite_without_warning():
    image = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    assert psnr(image, image) == float("inf")


def test_multiclass_overlap_metrics_returns_macro_and_per_class_scores():
    mask_true = np.array(
        [
            [0, 1, 1],
            [0, 2, 2],
            [0, 3, 3],
        ],
        dtype=np.int64,
    )
    mask_pred = mask_true.copy()
    metrics = multiclass_overlap_metrics(mask_true, mask_pred, class_values=(0, 1, 2, 3))
    assert metrics["class_1_dice"] == 1.0
    assert metrics["class_2_iou"] == 1.0
    assert metrics["class_3_hd95"] == 0.0
    assert metrics["macro_dice"] == 1.0
