from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
from skimage.measure import find_contours
from skimage.transform import resize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cardiac_image_system.core.io import load_grayscale_image, load_label_image
from cardiac_image_system.core.preprocessing import preprocess_image
from cardiac_image_system.core.torch_data import resize_label_map
from cardiac_image_system.models.unet2d import UNet2D


WORKSPACE_ROOT = PROJECT_ROOT.parent.parent
MANUSCRIPT_DIR = (
    WORKSPACE_ROOT
    / "05_Дополнительные_материалы"
    / "02_Черновики_и_рукописи"
    / "Radiology_AI_2026"
)
FIGURES_DIR = MANUSCRIPT_DIR / "figures"

ACDC_TEST_MANIFEST = PROJECT_ROOT / "data" / "manifests_local" / "acdc_test.csv"
CAMUS_TEST_MANIFEST = PROJECT_ROOT / "data" / "manifests_local" / "camus_test.csv"
MULTICLASS_RUN_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "unet_multiclass_acdc_top3_extended"
    / "acdc_multiclass_top3_extended_20260702_100036"
)
STATS_CSV = (
    WORKSPACE_ROOT
    / "05_Дополнительные_материалы"
    / "05_Технические_отчеты_и_реестры"
    / "radiology_ai_multiclass_acdc_top3_extended_stats_2026.csv"
)

MODE_LABELS = {
    "none": "Raw input",
    "wavelet": "Wavelet denoising",
    "nlm": "Nonlocal means",
}
MODE_COLORS = {
    "none": "#1b4965",
    "wavelet": "#ca6702",
    "nlm": "#2a9d8f",
}
CLASS_COLORS = {
    1: "#f4a261",  # RV
    2: "#e76f51",  # Myocardium
    3: "#2a9d8f",  # LV
}
CLASS_LABELS = {
    1: "RV cavity",
    2: "Myocardium",
    3: "LV cavity",
}
PRIMARY_EFFECT_DATA = {
    "ACDC": [
        {"mode": "gaussian", "delta": -0.0130, "ci_low": -0.0247, "ci_high": -0.0012, "p": 0.1110},
        {"mode": "wavelet", "delta": -0.0097, "ci_low": -0.0212, "ci_high": 0.0008, "p": 0.1110},
        {"mode": "nlm", "delta": -0.0115, "ci_low": -0.0224, "ci_high": -0.0017, "p": 0.1110},
        {"mode": "clahe", "delta": -0.0257, "ci_low": -0.0405, "ci_high": -0.0112, "p": 0.0118},
        {"mode": "hybrid", "delta": -0.0029, "ci_low": -0.0097, "ci_high": 0.0049, "p": 0.1110},
    ],
    "CAMUS": [
        {"mode": "gaussian", "delta": -0.0050, "ci_low": -0.0087, "ci_high": -0.0014, "p": 0.1088},
        {"mode": "wavelet", "delta": 0.0030, "ci_low": 0.0002, "ci_high": 0.0059, "p": 0.1269},
        {"mode": "nlm", "delta": 0.0012, "ci_low": -0.0012, "ci_high": 0.0035, "p": 0.4097},
        {"mode": "clahe", "delta": -0.0144, "ci_low": -0.0186, "ci_high": -0.0104, "p": 0.0001},
        {"mode": "hybrid", "delta": -0.0097, "ci_low": -0.0140, "ci_high": -0.0058, "p": 0.0001},
    ],
    "Mixed corpus": [
        {"mode": "gaussian", "delta": -0.0331, "ci_low": -0.0441, "ci_high": -0.0232, "p": 0.0001},
        {"mode": "wavelet", "delta": -0.0175, "ci_low": -0.0274, "ci_high": -0.0092, "p": 0.0001},
        {"mode": "nlm", "delta": -0.0055, "ci_low": -0.0102, "ci_high": -0.0011, "p": 0.0516},
        {"mode": "clahe", "delta": -0.0245, "ci_low": -0.0329, "ci_high": -0.0166, "p": 0.0001},
        {"mode": "hybrid", "delta": -0.0477, "ci_low": -0.0577, "ci_high": -0.0381, "p": 0.0001},
    ],
}
ROBUSTNESS_DATA = {
    "CAMUS": {
        "Primary": {
            "none": {"dice": 0.9180, "hd95": 13.1536},
            "wavelet": {"dice": 0.9210, "hd95": 13.5043},
            "nlm": {"dice": 0.9192, "hd95": 13.6298},
        },
        "Multiseed": {
            "none": {"dice": 0.9194, "hd95": 13.3362},
            "wavelet": {"dice": 0.9184, "hd95": 13.9153},
            "nlm": {"dice": 0.9142, "hd95": 14.4028},
        },
        "Long schedule": {
            "none": {"dice": 0.9330, "hd95": 10.4415},
            "wavelet": {"dice": 0.9288, "hd95": 11.6109},
            "nlm": {"dice": 0.9350, "hd95": 10.8893},
        },
    },
    "Mixed corpus": {
        "Primary": {
            "none": {"dice": 0.8951, "hd95": 13.5698},
            "wavelet": {"dice": 0.8776, "hd95": 14.8727},
            "nlm": {"dice": 0.8896, "hd95": 13.6845},
        },
        "Multiseed": {
            "none": {"dice": 0.8958, "hd95": 13.6381},
            "wavelet": {"dice": 0.8922, "hd95": 14.2170},
            "nlm": {"dice": 0.8976, "hd95": 12.9321},
        },
        "Long schedule": {
            "none": {"dice": 0.9167, "hd95": 9.2863},
            "wavelet": {"dice": 0.9152, "hd95": 11.3420},
            "nlm": {"dice": 0.9202, "hd95": 9.9780},
        },
    },
}


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "font.family": "DejaVu Serif",
        }
    )


def _save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{stem}.png", bbox_inches="tight", dpi=300)
    fig.savefig(FIGURES_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _select_representative_acdc_row() -> pd.Series:
    manifest = pd.read_csv(ACDC_TEST_MANIFEST)
    best_row = None
    best_area = -1
    for _, row in manifest.iterrows():
        mask = load_label_image(row["mask_path"], int(row["slice_index"]))
        unique = set(np.unique(mask).tolist())
        if not {1, 2, 3}.issubset(unique):
            continue
        area = int((mask > 0).sum())
        if area > best_area:
            best_area = area
            best_row = row
    if best_row is None:
        raise RuntimeError("Unable to identify a representative ACDC slice with all foreground classes.")
    return best_row


def _select_representative_camus_row() -> pd.Series:
    manifest = pd.read_csv(CAMUS_TEST_MANIFEST)
    best_row = None
    best_area = -1
    for _, row in manifest.iterrows():
        mask = load_label_image(row["mask_path"], int(row["slice_index"]))
        area = int((mask > 0).sum())
        if area > best_area:
            best_area = area
            best_row = row
    if best_row is None:
        raise RuntimeError("Unable to identify a representative CAMUS frame.")
    return best_row


def _resize_image_for_model(image: np.ndarray, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    return resize(
        image,
        size,
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)


def _load_model(mode: str) -> UNet2D:
    checkpoint_path = MULTICLASS_RUN_DIR / f"unet_multiclass_extended_acdc_{mode}" / "checkpoint_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = UNet2D(in_channels=1, out_channels=4)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _predict_multiclass_label_map(model: UNet2D, image_2d: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(image_2d[None, None, ...]).float()
    with torch.no_grad():
        logits = model(tensor)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int64)
    return pred


def _label_overlay(label_map: np.ndarray, alpha: float = 0.34) -> np.ndarray:
    overlay = np.zeros(label_map.shape + (4,), dtype=np.float32)
    for class_value, color in CLASS_COLORS.items():
        rgb = matplotlib.colors.to_rgb(color)
        mask = label_map == class_value
        overlay[mask, :3] = rgb
        overlay[mask, 3] = alpha
    return overlay


def _draw_contours(ax: plt.Axes, label_map: np.ndarray, linestyle: str, linewidth: float, alpha: float = 1.0) -> None:
    for class_value, color in CLASS_COLORS.items():
        binary = (label_map == class_value).astype(np.uint8)
        if binary.sum() == 0:
            continue
        for contour in find_contours(binary, 0.5):
            ax.plot(contour[:, 1], contour[:, 0], color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha)


def _draw_binary_contour(ax: plt.Axes, mask: np.ndarray, color: str = "#e9ecef", linewidth: float = 1.2) -> None:
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return
    for contour in find_contours(binary, 0.5):
        ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=linewidth)


def create_workflow_figure() -> None:
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def add_box(x: float, y: float, w: float, h: float, text: str, facecolor: str, edgecolor: str = "#2f3e46") -> tuple[float, float]:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.2,
            edgecolor=edgecolor,
            facecolor=facecolor,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", wrap=True)
        return x + w / 2, y + h / 2

    primary_color = "#d8f3dc"
    secondary_color = "#dbeafe"
    accent_color = "#ffe8cc"

    add_box(0.03, 0.66, 0.18, 0.18, "Public datasets\nACDC: 150 patients\nCAMUS: 500 patients", primary_color)
    add_box(0.27, 0.66, 0.18, 0.18, "Manifest construction\nPositive-mask slices/frames\nMetadata retained", secondary_color)
    add_box(0.51, 0.66, 0.18, 0.18, "Patient-level splitting\nTrain / validation / test\nLeakage audit", secondary_color)
    add_box(0.75, 0.66, 0.20, 0.18, "Primary binary benchmark\n6 preprocessing modes\nFixed 2D U-Net\nDice, IoU, HD95", accent_color)

    add_box(0.09, 0.28, 0.18, 0.16, "Top-3 ACDC follow-up\nnone, wavelet, nlm", primary_color)
    add_box(0.34, 0.28, 0.18, 0.16, "Class-resolved branch\nBackground, RV,\nmyocardium, LV", secondary_color)
    add_box(0.59, 0.28, 0.18, 0.16, "Long-schedule rerun\nUp to 50 epochs\nEarly stopping", secondary_color)
    add_box(0.81, 0.28, 0.15, 0.16, "Paired statistics\nWilcoxon + Holm\nBootstrap CIs", accent_color)

    arrowprops = dict(arrowstyle="-|>", lw=1.4, color="#344e41", shrinkA=6, shrinkB=6)
    ax.annotate("", xy=(0.27, 0.75), xytext=(0.21, 0.75), arrowprops=arrowprops)
    ax.annotate("", xy=(0.51, 0.75), xytext=(0.45, 0.75), arrowprops=arrowprops)
    ax.annotate("", xy=(0.75, 0.75), xytext=(0.69, 0.75), arrowprops=arrowprops)

    ax.annotate("", xy=(0.18, 0.44), xytext=(0.83, 0.66), arrowprops=arrowprops)
    ax.annotate("", xy=(0.34, 0.36), xytext=(0.27, 0.36), arrowprops=arrowprops)
    ax.annotate("", xy=(0.59, 0.36), xytext=(0.52, 0.36), arrowprops=arrowprops)
    ax.annotate("", xy=(0.81, 0.36), xytext=(0.77, 0.36), arrowprops=arrowprops)

    ax.text(
        0.03,
        0.93,
        "Implemented benchmark workflow and targeted robustness branch",
        fontsize=14,
        fontweight="bold",
        ha="left",
    )
    ax.text(
        0.03,
        0.08,
        "Primary branch: modality-aware public benchmarking with six preprocessing modes under a fixed U-Net baseline.\n"
        "Secondary branch: class-resolved ACDC follow-up added to test whether binary label collapse or short training concealed a preprocessing benefit.",
        fontsize=10,
        ha="left",
        va="bottom",
    )

    _save_figure(fig, "figure_1_benchmark_workflow")


def create_preprocessing_gallery_figure() -> None:
    acdc_row = _select_representative_acdc_row()
    camus_row = _select_representative_camus_row()
    modes = ["none", "gaussian", "wavelet", "nlm", "clahe", "hybrid"]
    titles = [
        "Raw input",
        "Gaussian",
        "Wavelet",
        "NLM",
        "CLAHE",
        "Hybrid",
    ]

    def prepare_row(row: pd.Series) -> tuple[list[np.ndarray], np.ndarray]:
        slice_index = int(row["slice_index"])
        image = load_grayscale_image(row["image_path"], slice_index=slice_index)
        mask = load_label_image(row["mask_path"], slice_index=slice_index)
        mask_resized = resize_label_map((mask > 0).astype(np.uint8), (256, 256))
        processed_images = []
        for mode in modes:
            processed = preprocess_image(image, mode=mode)
            processed_images.append(_resize_image_for_model(processed))
        return processed_images, mask_resized

    acdc_images, acdc_mask = prepare_row(acdc_row)
    camus_images, camus_mask = prepare_row(camus_row)

    fig, axes = plt.subplots(2, 6, figsize=(16, 6))
    for col, title in enumerate(titles):
        axes[0, col].set_title(title)

    row_payload = [
        ("ACDC cine CMR", acdc_images, acdc_mask),
        ("CAMUS echocardiography", camus_images, camus_mask),
    ]
    for row_idx, (row_label, images, mask) in enumerate(row_payload):
        for col_idx, image in enumerate(images):
            ax = axes[row_idx, col_idx]
            ax.imshow(image, cmap="gray")
            _draw_binary_contour(ax, mask, color="#f8f9fa", linewidth=1.1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[row_idx, 0].text(
            -0.18,
            0.5,
            row_label,
            transform=axes[row_idx, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    fig.suptitle(
        "Representative appearance of the six preprocessing modes across cine CMR and echocardiography",
        fontsize=12,
        y=0.98,
    )
    fig.text(
        0.5,
        0.02,
        "Reference binary contour shown in white to indicate the foreground region across modalities.",
        ha="center",
        va="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0.04, 0.05, 1, 0.94])
    _save_figure(fig, "figure_2_preprocessing_gallery")


def create_qualitative_figure() -> None:
    row = _select_representative_acdc_row()
    slice_index = int(row["slice_index"])
    raw_image = load_grayscale_image(row["image_path"], slice_index=slice_index)
    raw_mask = load_label_image(row["mask_path"], slice_index=slice_index)
    gt_labels = resize_label_map(raw_mask, (256, 256))

    backgrounds: dict[str, np.ndarray] = {}
    predictions: dict[str, np.ndarray] = {}
    for mode in ("none", "wavelet", "nlm"):
        processed = preprocess_image(raw_image, mode=mode)
        image_resized = _resize_image_for_model(processed)
        model = _load_model(mode)
        pred_labels = _predict_multiclass_label_map(model, image_resized)
        backgrounds[mode] = image_resized
        predictions[mode] = pred_labels

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
    panel_titles = [
        "Reference slice + ground truth",
        "Raw input mode",
        "Wavelet mode",
        "Nonlocal-means mode",
    ]

    axes[0].imshow(backgrounds["none"], cmap="gray")
    axes[0].imshow(_label_overlay(gt_labels, alpha=0.38))
    _draw_contours(axes[0], gt_labels, linestyle="-", linewidth=1.2)
    axes[0].set_title(panel_titles[0])

    for ax, mode, title in zip(axes[1:], ("none", "wavelet", "nlm"), panel_titles[1:]):
        ax.imshow(backgrounds[mode], cmap="gray")
        ax.imshow(_label_overlay(predictions[mode], alpha=0.34))
        _draw_contours(ax, gt_labels, linestyle="-", linewidth=1.0, alpha=0.95)
        _draw_contours(ax, predictions[mode], linestyle="--", linewidth=1.2, alpha=1.0)
        ax.set_title(title)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    legend_handles = [Patch(facecolor=matplotlib.colors.to_rgba(color, 0.45), edgecolor=color, label=label) for _, (color, label) in enumerate(zip(CLASS_COLORS.values(), CLASS_LABELS.values()))]
    contour_handles = [
        Line2D([0], [0], color="black", linewidth=1.2, linestyle="-", label="Reference contour"),
        Line2D([0], [0], color="black", linewidth=1.2, linestyle="--", label="Prediction contour"),
    ]
    fig.legend(
        legend_handles + contour_handles,
        [handle.get_label() for handle in legend_handles + contour_handles],
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    patient_label = str(row["patient_id"]).replace("ACDC_", "")
    fig.suptitle(
        f"Representative ACDC test slice ({patient_label}, {row['phase']}, slice {slice_index}) from the long-schedule multiclass top-3 follow-up",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    _save_figure(fig, "figure_2_acdc_multiclass_qualitative")


def create_followup_figure() -> None:
    histories = {}
    patient_level = {}
    best_epochs = {}

    for mode in ("none", "wavelet", "nlm"):
        run_dir = MULTICLASS_RUN_DIR / f"unet_multiclass_extended_acdc_{mode}"
        histories[mode] = pd.read_csv(run_dir / "history.csv")
        patient_level[mode] = pd.read_csv(run_dir / "test_patient_level.csv")
        summary = pd.read_json(run_dir / "summary.json", typ="series")
        best_epochs[mode] = int(summary["best_epoch"])

    stats_df = pd.read_csv(STATS_CSV)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax_curve = axes[0]
    for mode in ("none", "wavelet", "nlm"):
        hist = histories[mode]
        ax_curve.plot(hist["epoch"], hist["val_loss"], label=MODE_LABELS[mode], color=MODE_COLORS[mode], linewidth=2.0)
        best_epoch = best_epochs[mode]
        best_row = hist.loc[hist["epoch"] == best_epoch].iloc[0]
        ax_curve.scatter(best_epoch, best_row["val_loss"], color=MODE_COLORS[mode], s=40, zorder=5)
        ax_curve.annotate(
            f"best={best_epoch}",
            (best_epoch, best_row["val_loss"]),
            textcoords="offset points",
            xytext=(0, -12),
            ha="center",
            color=MODE_COLORS[mode],
            fontsize=8,
        )
    ax_curve.set_title("A. Validation-loss trajectories in the long-schedule ACDC follow-up")
    ax_curve.set_xlabel("Epoch")
    ax_curve.set_ylabel("Validation loss")
    ax_curve.grid(True, alpha=0.25)
    ax_curve.legend(frameon=False, loc="upper right")

    ax_bar = axes[1]
    metrics = ["macro_dice", "class_1_dice", "class_2_dice", "class_3_dice"]
    metric_labels = ["Macro", "RV", "Myocardium", "LV"]
    x = np.arange(len(metrics))
    width = 0.24
    offsets = {"none": -width, "wavelet": 0.0, "nlm": width}
    for mode in ("none", "wavelet", "nlm"):
        means = [float(patient_level[mode][metric].mean()) for metric in metrics]
        stds = [float(patient_level[mode][metric].std(ddof=1)) for metric in metrics]
        ax_bar.bar(
            x + offsets[mode],
            means,
            width=width,
            yerr=stds,
            capsize=3,
            label=MODE_LABELS[mode],
            color=MODE_COLORS[mode],
            alpha=0.9,
        )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_labels)
    ax_bar.set_ylim(0.68, 0.95)
    ax_bar.set_ylabel("Patient-level Dice")
    ax_bar.set_title("B. Class-resolved Dice summary")
    ax_bar.grid(axis="y", alpha=0.25)

    macro_wavelet = stats_df[(stats_df["metric"] == "Macro Dice") & (stats_df["mode"] == "wavelet")].iloc[0]
    macro_nlm = stats_df[(stats_df["metric"] == "Macro Dice") & (stats_df["mode"] == "nlm")].iloc[0]
    text = (
        "Macro Dice vs raw input:\n"
        f"Wavelet: Δ={macro_wavelet['delta_vs_none']:.4f}, Holm p={macro_wavelet['p_holm']:.4f}\n"
        f"NLM: Δ={macro_nlm['delta_vs_none']:.4f}, Holm p={macro_nlm['p_holm']:.4f}"
    )
    ax_bar.text(
        0.02,
        0.02,
        text,
        transform=ax_bar.transAxes,
        fontsize=8.5,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#bbbbbb", alpha=0.95),
    )

    fig.suptitle("Long-schedule targeted multiclass ACDC follow-up", fontsize=12, y=1.02)
    fig.tight_layout()
    _save_figure(fig, "figure_3_acdc_multiclass_followup")


def create_primary_effect_figure() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6), sharex=True)
    mode_order = ["gaussian", "wavelet", "nlm", "clahe", "hybrid"]
    mode_labels = {
        "gaussian": "Gaussian",
        "wavelet": "Wavelet",
        "nlm": "NLM",
        "clahe": "CLAHE",
        "hybrid": "Hybrid",
    }
    dataset_titles = list(PRIMARY_EFFECT_DATA.keys())
    for ax, dataset_title in zip(axes, dataset_titles):
        rows = PRIMARY_EFFECT_DATA[dataset_title]
        y = np.arange(len(mode_order))
        ax.axvline(0.0, color="#6c757d", linestyle="--", linewidth=1.1)
        for idx, mode in enumerate(mode_order):
            row = next(item for item in rows if item["mode"] == mode)
            significant = row["p"] < 0.05
            if significant and row["delta"] < 0:
                color = "#b02a37"
            elif significant and row["delta"] > 0:
                color = "#0c7c59"
            else:
                color = "#495057"
            ax.plot([row["ci_low"], row["ci_high"]], [idx, idx], color=color, linewidth=2.4)
            ax.scatter(row["delta"], idx, color=color, s=42, zorder=3)
            p_label = "<0.0001" if row["p"] <= 0.0001 else f"{row['p']:.4f}"
            ax.text(
                row["ci_high"] + 0.0015,
                idx,
                f"p={p_label}",
                va="center",
                ha="left",
                fontsize=8,
                color=color,
            )
        ax.set_yticks(y)
        ax.set_yticklabels([mode_labels[mode] for mode in mode_order])
        ax.invert_yaxis()
        ax.set_title(dataset_title)
        ax.grid(axis="x", alpha=0.22)
        ax.set_xlim(-0.065, 0.015)
        ax.set_xlabel("Mean Dice difference vs raw input")
    axes[0].set_ylabel("Preprocessing mode")
    fig.suptitle(
        "Patient-level Dice shifts relative to the raw-input baseline with 95% bootstrap confidence intervals",
        fontsize=12,
        y=0.98,
    )
    fig.text(
        0.5,
        0.02,
        "Vertical dashed line marks no difference. Red indicates statistically supported degradation after Holm correction.",
        ha="center",
        va="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    _save_figure(fig, "figure_5_primary_effects_forest")


def create_binary_robustness_synthesis_figure() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex="col")
    branches = ["Primary", "Multiseed", "Long schedule"]
    x = np.arange(len(branches))
    dataset_panels = [("CAMUS", 0), ("Mixed corpus", 1)]

    for dataset_name, col_idx in dataset_panels:
        dice_ax = axes[0, col_idx]
        hd95_ax = axes[1, col_idx]
        dataset = ROBUSTNESS_DATA[dataset_name]
        for mode in ("none", "wavelet", "nlm"):
            dice_vals = [dataset[branch][mode]["dice"] for branch in branches]
            hd95_vals = [dataset[branch][mode]["hd95"] for branch in branches]
            color = MODE_COLORS[mode]
            label = MODE_LABELS[mode]
            dice_ax.plot(x, dice_vals, marker="o", linewidth=2.0, color=color, label=label)
            hd95_ax.plot(x, hd95_vals, marker="o", linewidth=2.0, color=color, label=label)

        dice_ax.set_title(f"{dataset_name}: Dice across robustness branches")
        dice_ax.set_ylabel("Patient-level Dice")
        dice_ax.set_xticks(x)
        dice_ax.set_xticklabels(branches)
        dice_ax.grid(True, alpha=0.25)

        hd95_ax.set_title(f"{dataset_name}: HD95 across robustness branches")
        hd95_ax.set_ylabel("Patient-level HD95")
        hd95_ax.set_xticks(x)
        hd95_ax.set_xticklabels(branches)
        hd95_ax.grid(True, alpha=0.25)

    axes[0, 0].legend(frameon=False, loc="lower right")
    fig.suptitle(
        "Top-3 binary robustness synthesis for CAMUS and the mixed-corpus benchmark",
        fontsize=12,
        y=0.98,
    )
    fig.text(
        0.5,
        0.02,
        "Across follow-up branches, small Dice rank shifts do not translate into a stable, decision-relevant preprocessing advantage.",
        ha="center",
        va="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    _save_figure(fig, "figure_6_binary_robustness_synthesis")


def main() -> None:
    _configure_style()
    create_workflow_figure()
    create_preprocessing_gallery_figure()
    create_qualitative_figure()
    create_followup_figure()
    create_primary_effect_figure()
    create_binary_robustness_synthesis_figure()
    print(f"Saved figure package to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
