from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cardiac_image_system.core.manifest import load_manifest, summarize_manifest
from cardiac_image_system.core.metrics import dice, dice_from_counts, hd95, iou, iou_from_counts, overlap_counts
from cardiac_image_system.core.preprocessing import MmSpaceFilterTargets, PreprocessParams
from cardiac_image_system.core.splits import (
    export_split_manifests,
    make_patient_level_random_split,
    split_by_subset_column,
    split_by_subset_column_carving_validation,
)
from cardiac_image_system.core.torch_data import ManifestSegmentationDataset
from cardiac_image_system.core.validation import (
    aggregate_patient_level,
    aggregate_volumetric_level,
    save_runtime_log,
    validate_patient_level_split,
)
from cardiac_image_system.models import AttentionUNet2D, TransUNet2D, UNet2D

ARCHITECTURES = {"unet": UNet2D, "attention_unet": AttentionUNet2D, "transunet": TransUNet2D}


@dataclass(frozen=True)
class TrainConfig:
    architecture: str = "unet"
    preprocess_mode: str = "none"
    image_height: int = 256
    image_width: int = 256
    batch_size: int = 8
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    seed: int = 42
    validation_seed: int = 42
    threshold: float = 0.5
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    max_test_samples: int | None = None
    dataset_filter: tuple[str, ...] = ()
    early_stopping_patience: int = 0
    early_stopping_min_epochs: int = 0
    early_stopping_min_delta: float = 0.0
    wavelet_level: int = 2
    nlm_h_multiplier: float = 0.8
    clahe_clip_limit: float = 0.03
    preprocess_cache_dir: str | None = None
    gaussian_sigma_mm: float | None = None
    nlm_patch_size_mm: float | None = None
    nlm_patch_distance_mm: float | None = None
    clahe_kernel_size_mm: float | None = None

    def mm_space_targets(self) -> MmSpaceFilterTargets | None:
        if (
            self.gaussian_sigma_mm is None
            and self.nlm_patch_size_mm is None
            and self.nlm_patch_distance_mm is None
            and self.clahe_kernel_size_mm is None
        ):
            return None
        return MmSpaceFilterTargets(
            gaussian_sigma_mm=self.gaussian_sigma_mm,
            nlm_patch_size_mm=self.nlm_patch_size_mm,
            nlm_patch_distance_mm=self.nlm_patch_distance_mm,
            clahe_kernel_size_mm=self.clahe_kernel_size_mm,
        )

    def preprocess_params(self) -> PreprocessParams:
        return PreprocessParams(
            wavelet_level=self.wavelet_level,
            nlm_h_multiplier=self.nlm_h_multiplier,
            clahe_clip_limit=self.clahe_clip_limit,
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice_score = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice_score.mean()


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    dice_term = dice_loss_from_logits(logits, targets)
    return 0.5 * bce + 0.5 * dice_term


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _limit_df(df: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None or len(df) <= limit:
        return df.copy()
    return df.iloc[:limit].copy()


def apply_dataset_filter(df: pd.DataFrame, dataset_filter: tuple[str, ...]) -> pd.DataFrame:
    if not dataset_filter or "dataset" not in df.columns:
        return df.copy()
    filtered = df[df["dataset"].astype(str).isin(dataset_filter)].copy()
    if filtered.empty:
        raise ValueError("No rows left in manifest after dataset filtering")
    return filtered


def resolve_split_map(manifest: pd.DataFrame, config: TrainConfig) -> dict[str, pd.DataFrame]:
    df = apply_dataset_filter(manifest, config.dataset_filter)
    stratify_by = [col for col in ["dataset", "group", "view"] if col in df.columns]

    try:
        subset_split = split_by_subset_column(df)
        if not subset_split["train"].empty and not subset_split["val"].empty and not subset_split["test"].empty:
            return subset_split
    except ValueError:
        pass

    try:
        return split_by_subset_column_carving_validation(
            df,
            val_ratio=config.val_ratio,
            validation_seed=config.validation_seed,
            stratify_by=stratify_by,
        )
    except ValueError:
        pass

    return make_patient_level_random_split(
        df,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
        stratify_by=stratify_by,
    )


def build_dataloader(
    manifest: pd.DataFrame,
    config: TrainConfig,
    augment: bool,
    shuffle: bool,
) -> DataLoader:
    dataset = ManifestSegmentationDataset(
        manifest=manifest,
        image_size=(config.image_height, config.image_width),
        preprocess_mode=config.preprocess_mode,
        preprocess_params=config.preprocess_params(),
        augment=augment,
        seed=config.seed,
        preprocess_cache_dir=config.preprocess_cache_dir,
        mm_space_targets=config.mm_space_targets(),
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=False,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
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


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    mode: str,
) -> tuple[float, list[dict[str, object]], float]:
    model.eval()
    losses: list[float] = []
    rows: list[dict[str, object]] = []
    inference_times_ms: list[float] = []

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
                        "mode": mode,
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

    mean_loss = float(np.mean(losses)) if losses else float("nan")
    mean_inference_ms = float(np.mean(inference_times_ms)) if inference_times_ms else float("nan")
    return mean_loss, rows, mean_inference_ms


def save_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--architecture", choices=sorted(ARCHITECTURES), default="unet")
    parser.add_argument("--mode", default="none")
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
    parser.add_argument("--wavelet-level", type=int, default=2)
    parser.add_argument("--nlm-h-multiplier", type=float, default=0.8)
    parser.add_argument("--clahe-clip-limit", type=float, default=0.03)
    parser.add_argument("--preprocess-cache-dir", type=str, default=None)
    parser.add_argument("--gaussian-sigma-mm", type=float, default=None)
    parser.add_argument("--nlm-patch-size-mm", type=float, default=None)
    parser.add_argument("--nlm-patch-distance-mm", type=float, default=None)
    parser.add_argument("--clahe-kernel-size-mm", type=float, default=None)
    args = parser.parse_args()

    config = TrainConfig(
        architecture=args.architecture,
        preprocess_mode=args.mode,
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
        wavelet_level=args.wavelet_level,
        nlm_h_multiplier=args.nlm_h_multiplier,
        clahe_clip_limit=args.clahe_clip_limit,
        preprocess_cache_dir=args.preprocess_cache_dir,
        gaussian_sigma_mm=args.gaussian_sigma_mm,
        nlm_patch_size_mm=args.nlm_patch_size_mm,
        nlm_patch_distance_mm=args.nlm_patch_distance_mm,
        clahe_kernel_size_mm=args.clahe_kernel_size_mm,
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

    export_split_manifests(split_map, output_dir / "resolved_splits", prefix="unet_baseline")

    train_loader = build_dataloader(split_map["train"], config=config, augment=True, shuffle=True)
    val_loader = build_dataloader(split_map["val"], config=config, augment=False, shuffle=False)
    test_loader = build_dataloader(split_map["test"], config=config, augment=False, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ARCHITECTURES[config.architecture]().to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(config.epochs, 2))

    log(f"Starting baseline run: architecture={config.architecture}, mode={config.preprocess_mode}, output_dir={output_dir}")
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

    history_rows: list[dict[str, object]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    completed_epochs = 0
    stopped_early = False
    epochs_without_improvement = 0
    best_checkpoint_path = output_dir / "checkpoint_best.pt"

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer=optimizer, device=device)
        val_loss, _, val_inference_ms = evaluate_model(
            model,
            val_loader,
            device=device,
            threshold=config.threshold,
            mode=config.preprocess_mode,
        )
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
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                },
                best_checkpoint_path,
            )
            log(f"Epoch {epoch:03d}: new best checkpoint saved with val_loss={val_loss:.6f}")
        else:
            epochs_without_improvement += 1

        log(
            f"Epoch {epoch:03d}/{config.epochs}: "
            f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
            f"lr={optimizer.param_groups[0]['lr']:.6e}, "
            f"val_inference_ms={val_inference_ms:.2f}"
        )
        completed_epochs = epoch

        if (
            config.early_stopping_patience > 0
            and epoch >= max(config.early_stopping_min_epochs, 1)
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            stopped_early = True
            log(
                "Early stopping triggered: "
                f"no validation-loss improvement greater than {config.early_stopping_min_delta:.6f} "
                f"for {epochs_without_improvement} epoch(s)."
            )
            break

    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        log(f"Loaded best checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    test_loss, test_rows, test_inference_ms = evaluate_model(
        model,
        test_loader,
        device=device,
        threshold=config.threshold,
        mode=config.preprocess_mode,
    )
    test_slice_df = pd.DataFrame(test_rows)
    test_slice_df.to_csv(output_dir / "test_slice_level.csv", index=False)
    patient_df = aggregate_patient_level(test_slice_df)
    patient_df.to_csv(output_dir / "test_patient_level.csv", index=False)
    volumetric_df = aggregate_volumetric_level(test_slice_df)
    volumetric_df.to_csv(output_dir / "test_volumetric_level.csv", index=False)

    summary = {
        "config": asdict(config),
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
        "test_metrics_slice_mean": {
            "dice": float(test_slice_df["dice"].mean()) if not test_slice_df.empty else float("nan"),
            "iou": float(test_slice_df["iou"].mean()) if not test_slice_df.empty else float("nan"),
            "hd95": float(test_slice_df["hd95"].replace([np.inf, -np.inf], np.nan).mean()) if not test_slice_df.empty else float("nan"),
        },
        "test_metrics_patient_mean": {
            "dice": float(patient_df["dice"].mean()) if not patient_df.empty else float("nan"),
            "iou": float(patient_df["iou"].mean()) if not patient_df.empty else float("nan"),
            "hd95": float(patient_df["hd95"].replace([np.inf, -np.inf], np.nan).mean()) if not patient_df.empty else float("nan"),
            "hd95_mm": (
                float(patient_df["hd95_mm"].replace([np.inf, -np.inf], np.nan).dropna().mean())
                if not patient_df.empty and "hd95_mm" in patient_df.columns and patient_df["hd95_mm"].notna().any()
                else float("nan")
            ),
        },
        "test_metrics_volumetric_mean": {
            "dice_3d": float(volumetric_df["dice_3d"].mean()) if not volumetric_df.empty else float("nan"),
            "iou_3d": float(volumetric_df["iou_3d"].mean()) if not volumetric_df.empty else float("nan"),
        },
    }
    save_json(output_dir / "summary.json", summary)
    log(
        "Test summary: "
        f"loss={test_loss:.6f}, "
        f"dice={summary['test_metrics_slice_mean']['dice']:.6f}, "
        f"iou={summary['test_metrics_slice_mean']['iou']:.6f}, "
        f"hd95={summary['test_metrics_slice_mean']['hd95']:.6f}, "
        f"inference_ms={test_inference_ms:.2f}"
    )

    save_runtime_log(
        output_dir,
        {
            "experiment": "train_unet_baseline",
            "mode": config.preprocess_mode,
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
