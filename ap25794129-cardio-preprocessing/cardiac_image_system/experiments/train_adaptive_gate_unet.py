"""Train AdaptiveGateUNet: a small learned gate network that mixes the candidate
preprocessing modes (none/gaussian/wavelet/nlm/clahe by default) per image, jointly
with a compact U-Net, on the segmentation loss itself.

This is the learned counterpart to every hand-picked Pθ composition evaluated
elsewhere in this benchmark (Gaussian, Wavelet, NLM, CLAHE, Hybrid, and their
mm-calibrated variants): instead of a fixed, dataset-wide operator, the effective
Pθ becomes image-conditional and is optimized directly against Dice/BCE, rather
than against an image-quality heuristic that must then be validated indirectly.

Reuses TrainConfig, resolve_split_map, and the patient-level split/leakage-check
machinery from train_unet_baseline.py unchanged, so this experiment sits on exactly
the same evaluation protocol (paired Wilcoxon + Holm + bootstrap CI + TOST via
compare_preprocessing_modes.py) as every other mode in this benchmark.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cardiac_image_system.core.manifest import load_manifest, summarize_manifest
from cardiac_image_system.core.metrics import hd95, overlap_counts, dice, iou
from cardiac_image_system.core.preprocessing import PreprocessParams
from cardiac_image_system.core.splits import export_split_manifests
from cardiac_image_system.core.torch_data import AdaptiveGateSegmentationDataset
from cardiac_image_system.core.validation import (
    aggregate_patient_level,
    aggregate_volumetric_level,
    save_runtime_log,
    validate_patient_level_split,
)
from cardiac_image_system.experiments.train_unet_baseline import (
    TrainConfig,
    apply_dataset_filter,
    count_trainable_parameters,
    log,
    resolve_split_map,
    save_json,
    set_seed,
    _limit_df,
)
from cardiac_image_system.models import AdaptiveGateUNet

DEFAULT_MODES = ("none", "gaussian", "wavelet", "nlm", "clahe")


def segmentation_loss(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, masks)
    probs = torch.sigmoid(logits)
    intersection = (probs * masks).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    dice_loss = 1.0 - ((2.0 * intersection + 1e-6) / (union + 1e-6))
    return 0.5 * bce + 0.5 * dice_loss.mean()


def build_gate_dataloader(
    manifest: pd.DataFrame,
    modes: tuple[str, ...],
    config: TrainConfig,
    augment: bool,
    shuffle: bool,
) -> DataLoader:
    dataset = AdaptiveGateSegmentationDataset(
        manifest=manifest,
        modes=modes,
        image_size=(config.image_height, config.image_width),
        preprocess_params=config.preprocess_params(),
        augment=augment,
        seed=config.seed,
        preprocess_cache_dir=config.preprocess_cache_dir,
    )
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_workers, pin_memory=False)


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    losses = []
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = segmentation_loss(logits, masks)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def evaluate_model(model, loader, device, threshold, modes):
    model.eval()
    losses = []
    rows = []
    gate_rows = []
    inference_times_ms = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            start = time.perf_counter()
            logits = model(images)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            inference_times_ms.append(elapsed_ms / max(int(images.size(0)), 1))

            loss = segmentation_loss(logits, masks)
            losses.append(float(loss.item()))

            probs = torch.sigmoid(logits).cpu().numpy()
            true_masks = masks.cpu().numpy()
            weights = model.gate_weights()
            weights_np = weights.cpu().numpy() if weights is not None else None

            spacing_rows = batch.get("resized_spacing_row_mm")
            spacing_cols = batch.get("resized_spacing_col_mm")

            batch_size = probs.shape[0]
            for index in range(batch_size):
                pred_mask = probs[index, 0] >= threshold
                true_mask = true_masks[index, 0] >= 0.5
                counts = overlap_counts(true_mask, pred_mask)

                spacing_row = float(spacing_rows[index]) if spacing_rows is not None else float("nan")
                spacing_col = float(spacing_cols[index]) if spacing_cols is not None else float("nan")
                hd95_mm = (
                    hd95(true_mask, pred_mask, spacing=(spacing_row, spacing_col))
                    if np.isfinite(spacing_row) and np.isfinite(spacing_col)
                    else float("nan")
                )

                rows.append(
                    {
                        "patient_id": batch["patient_id"][index],
                        "mode": "adaptive_gate",
                        "phase": batch["phase"][index],
                        "view": batch["view"][index],
                        "dataset": batch["dataset"][index],
                        "subset": batch["subset"][index],
                        "dice": dice(true_mask, pred_mask),
                        "iou": iou(true_mask, pred_mask),
                        "hd95": hd95(true_mask, pred_mask),
                        "hd95_mm": hd95_mm,
                        "spacing_row_mm": spacing_row,
                        "spacing_col_mm": spacing_col,
                        "foreground_pixels_true": int(true_mask.sum()),
                        "foreground_pixels_pred": int(pred_mask.sum()),
                        "intersection_pixels": counts["intersection"],
                    }
                )
                if weights_np is not None:
                    gate_row = {"patient_id": batch["patient_id"][index]}
                    gate_row.update({f"weight_{m}": float(weights_np[index, k]) for k, m in enumerate(modes)})
                    gate_rows.append(gate_row)

    mean_loss = float(np.mean(losses)) if losses else float("nan")
    mean_inference_ms = float(np.mean(inference_times_ms)) if inference_times_ms else float("nan")
    return mean_loss, rows, gate_rows, mean_inference_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--modes", type=str, default=",".join(DEFAULT_MODES))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--dataset-filter", nargs="*", default=[])
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-epochs", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--preprocess-cache-dir", type=str, default=None)
    args = parser.parse_args()

    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())

    config = TrainConfig(
        architecture="unet",
        preprocess_mode="none",  # unused by the gate dataset, kept for resolve_split_map's config type
        image_height=args.image_height,
        image_width=args.image_width,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        validation_seed=args.validation_seed,
        threshold=args.threshold,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        dataset_filter=tuple(args.dataset_filter),
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_epochs=args.early_stopping_min_epochs,
        early_stopping_min_delta=args.early_stopping_min_delta,
        preprocess_cache_dir=args.preprocess_cache_dir,
    )

    set_seed(config.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.manifest)
    effective_manifest = apply_dataset_filter(manifest, config.dataset_filter)
    split_map = resolve_split_map(effective_manifest, config=config)
    validate_patient_level_split(split_map["train"], split_map["val"], split_map["test"])

    split_map["train"] = _limit_df(split_map["train"], config.max_train_samples)
    split_map["val"] = _limit_df(split_map["val"], config.max_val_samples)
    split_map["test"] = _limit_df(split_map["test"], config.max_test_samples)

    export_split_manifests(split_map, output_dir / "resolved_splits", prefix="adaptive_gate")

    train_loader = build_gate_dataloader(split_map["train"], modes, config, augment=True, shuffle=True)
    val_loader = build_gate_dataloader(split_map["val"], modes, config, augment=False, shuffle=False)
    test_loader = build_gate_dataloader(split_map["test"], modes, config, augment=False, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdaptiveGateUNet(num_modes=len(modes)).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(config.epochs, 2))

    log(f"Starting adaptive-gate run: modes={modes}, output_dir={output_dir}")
    log(
        "Split summary: "
        f"train={len(split_map['train'])} rows/{split_map['train']['patient_id'].nunique()} patients, "
        f"val={len(split_map['val'])} rows/{split_map['val']['patient_id'].nunique()} patients, "
        f"test={len(split_map['test'])} rows/{split_map['test']['patient_id'].nunique()} patients"
    )
    log(
        f"Device={device}, parameters={count_trainable_parameters(model)}, "
        f"batch_size={config.batch_size}, epochs={config.epochs}, lr={config.learning_rate}"
    )

    history_rows = []
    best_val_loss = float("inf")
    best_epoch = 0
    completed_epochs = 0
    stopped_early = False
    epochs_without_improvement = 0
    best_checkpoint_path = output_dir / "checkpoint_best.pt"

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer=optimizer, device=device)
        val_loss, _, _, val_inference_ms = evaluate_model(model, val_loader, device=device, threshold=config.threshold, modes=modes)
        scheduler.step()

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "val_inference_ms_per_image": val_inference_ms,
                "epoch_seconds": time.perf_counter() - epoch_start,
            }
        )
        pd.DataFrame(history_rows).to_csv(output_dir / "history.csv", index=False)

        improvement = best_val_loss - val_loss
        if val_loss < best_val_loss and improvement > config.early_stopping_min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "modes": modes, "epoch": epoch, "best_val_loss": best_val_loss},
                best_checkpoint_path,
            )
            log(f"Epoch {epoch:03d}: new best checkpoint saved with val_loss={val_loss:.6f}")
        else:
            epochs_without_improvement += 1

        log(
            f"Epoch {epoch:03d}/{config.epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
            f"lr={optimizer.param_groups[0]['lr']:.6e}, val_inference_ms={val_inference_ms:.2f}"
        )
        completed_epochs = epoch

        if (
            config.early_stopping_patience > 0
            and epoch >= max(config.early_stopping_min_epochs, 1)
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            stopped_early = True
            log(f"Early stopping triggered after {epochs_without_improvement} epoch(s) without improvement.")
            break

    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        log(f"Loaded best checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    test_loss, test_rows, gate_rows, test_inference_ms = evaluate_model(
        model, test_loader, device=device, threshold=config.threshold, modes=modes
    )
    test_slice_df = pd.DataFrame(test_rows)
    test_slice_df.to_csv(output_dir / "test_slice_level.csv", index=False)
    patient_df = aggregate_patient_level(test_slice_df)
    patient_df.to_csv(output_dir / "test_patient_level.csv", index=False)
    volumetric_df = aggregate_volumetric_level(test_slice_df)
    volumetric_df.to_csv(output_dir / "test_volumetric_level.csv", index=False)

    gate_df = pd.DataFrame(gate_rows)
    gate_df.to_csv(output_dir / "test_gate_weights_slice_level.csv", index=False)
    gate_patient_mean = {}
    if not gate_df.empty:
        gate_patient_df = gate_df.groupby("patient_id", as_index=False).mean(numeric_only=True)
        gate_patient_df.to_csv(output_dir / "test_gate_weights_patient_level.csv", index=False)
        gate_patient_mean = {
            m: float(gate_patient_df[f"weight_{m}"].mean()) for m in modes if f"weight_{m}" in gate_patient_df.columns
        }

    summary = {
        "modes": list(modes),
        "device": str(device),
        "parameter_count": int(count_trainable_parameters(model)),
        "manifest_summary": summarize_manifest(effective_manifest),
        "train_rows": int(len(split_map["train"])),
        "val_rows": int(len(split_map["val"])),
        "test_rows": int(len(split_map["test"])),
        "train_patients": int(split_map["train"]["patient_id"].nunique()),
        "val_patients": int(split_map["val"]["patient_id"].nunique()),
        "test_patients": int(split_map["test"]["patient_id"].nunique()),
        "completed_epochs": int(completed_epochs),
        "best_epoch": int(best_epoch),
        "stopped_early": bool(stopped_early),
        "best_val_loss": float(best_val_loss),
        "test_loss": float(test_loss),
        "test_inference_ms_per_image": float(test_inference_ms),
        "test_metrics_patient_mean": {
            "dice": float(patient_df["dice"].mean()) if not patient_df.empty else float("nan"),
            "iou": float(patient_df["iou"].mean()) if not patient_df.empty else float("nan"),
            "hd95": float(patient_df["hd95"].replace([np.inf, -np.inf], np.nan).mean()) if not patient_df.empty else float("nan"),
        },
        "test_metrics_volumetric_mean": {
            "dice_3d": float(volumetric_df["dice_3d"].mean()) if not volumetric_df.empty else float("nan"),
            "iou_3d": float(volumetric_df["iou_3d"].mean()) if not volumetric_df.empty else float("nan"),
        },
        "mean_gate_weight_by_mode": gate_patient_mean,
    }
    save_json(output_dir / "summary.json", summary)
    log(
        f"Test summary: loss={test_loss:.6f}, dice={summary['test_metrics_patient_mean']['dice']:.6f}, "
        f"mean_gate_weights={gate_patient_mean}"
    )

    save_runtime_log(
        output_dir,
        {
            "experiment": "train_adaptive_gate_unet",
            "modes": list(modes),
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "device": str(device),
            "parameter_count": count_trainable_parameters(model),
            "manifest": str(args.manifest),
            "output_dir": str(output_dir),
            "completed_epochs": completed_epochs,
            "best_epoch": best_epoch,
            "stopped_early": stopped_early,
        },
    )
    log("Run completed successfully.")


if __name__ == "__main__":
    main()
