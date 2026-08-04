import numpy as np

from cardiac_image_system.core.metrics import (
    dice,
    dice_from_counts,
    hd95,
    iou,
    iou_from_counts,
    multiclass_overlap_metrics,
    overlap_counts,
    psnr,
    relative_area_error,
)


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


def test_hd95_with_anisotropic_spacing_scales_distance():
    true = np.zeros((10, 10), dtype=bool)
    pred = np.zeros((10, 10), dtype=bool)
    true[5, 5] = True
    pred[5, 8] = True  # 3 pixels away along the column axis only

    pixel_space = hd95(true, pred)
    assert pixel_space == 3.0

    # column spacing of 2.0mm should scale the (purely column-axis) distance by 2x
    mm_space = hd95(true, pred, spacing=(1.0, 2.0))
    assert mm_space == 6.0


def test_overlap_counts_pooled_across_slices_matches_full_volume_dice():
    rng = np.random.default_rng(0)
    slices_true = [rng.integers(0, 2, size=(6, 6)).astype(bool) for _ in range(4)]
    slices_pred = [rng.integers(0, 2, size=(6, 6)).astype(bool) for _ in range(4)]

    intersection = sum(overlap_counts(t, p)["intersection"] for t, p in zip(slices_true, slices_pred))
    true_area = sum(overlap_counts(t, p)["true_area"] for t, p in zip(slices_true, slices_pred))
    pred_area = sum(overlap_counts(t, p)["pred_area"] for t, p in zip(slices_true, slices_pred))

    pooled_dice = dice_from_counts(intersection, true_area, pred_area)
    pooled_iou = iou_from_counts(intersection, true_area, pred_area)

    full_volume_true = np.stack(slices_true)
    full_volume_pred = np.stack(slices_pred)
    assert pooled_dice == dice(full_volume_true, full_volume_pred)
    assert pooled_iou == iou(full_volume_true, full_volume_pred)

    mean_of_slice_dice = np.mean([dice(t, p) for t, p in zip(slices_true, slices_pred)])
    assert pooled_dice != mean_of_slice_dice  # pooled (volumetric) Dice is not the same statistic


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
