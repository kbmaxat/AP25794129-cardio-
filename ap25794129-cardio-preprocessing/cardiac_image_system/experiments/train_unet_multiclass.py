from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cardiac_image_system.core.manifest import load_manifest, summarize_manifest
from cardiac_image_system.core.metrics import multiclass_overlap_metrics
from cardiac_image_system.core.splits import export_split_manifests, make_patient_level_random_split, split_by_subset_column
from cardiac_image_system.core.torch_data import ManifestSegmentationDataset
from cardiac_image_system.core.validation import aggregate_patient_level, save_runtime_log, validate_patient_level_split
from cardiac_image_system.models import UNet2D


@dataclass(frozen=True)
class TrainConfig:
    preprocess_mode: str = "none"
    image_height: int = 256
    image_width: int = 256
    batch_size: int = 8
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    seed: int = 42
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    max_test_samples: int | None = None
    dataset_filter: tuple[str, ...] = ()
    class_values: tuple[int, ...] = (0, 1, 2, 3)
    class_names: tuple[str, ...] = ("background", "rv", "myocardium", "lv")
    ignore_background_in_dice: bool = True
    early_stopping_patience: int = 0
    early_stopping_min_epochs: int = 0
    early_stopping_min_delta: float = 0.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def multiclass_dice_loss_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_background: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    if ignore_background and probs.size(1) > 1:
        probs = probs[:, 1:, ...]
        targets = targets[:, 1:, ...]
    inter = (probs * targets).sum(dim=(2, 3))
    denom = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice_score = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice_score.mean()


def segmentation_loss(
    logits: torch.Tensor,
    target_masks: torch.Tensor,
    target_labels: torch.Tensor,
    ignore_background: bool = True,
) -> torch.Tensor:
    ce = nn.functional.cross_entropy(logits, target_labels)
    dice_term = multiclass_dice_loss_from_logits(
        logits,
        target_masks,
        ignore_background=ignore_background,
    )
    return 0.5 * ce + 0.5 * dice_term


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
    try:
        subset_split = split_by_subset_column(df)
        if not subset_split["train"].empty and not subset_split["val"].empty and not subset_split["test"].empty:
            return subset_split
    except ValueError:
        pass

    stratify_by = [col for col in ["dataset", "group", "view"] if col in df.columns]
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
        augment=augment,
        seed=config.seed,
        label_mode="multiclass",
        class_values=config.class_values,
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
    ignore_background_in_dice: bool,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        mask_labels = batch["mask_labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = segmentation_loss(
            logits,
            masks,
            mask_labels,
            ignore_background=ignore_background_in_dice,
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mode: str,
    class_values: tuple[int, ...],
    ignore_background_in_dice: bool,
) -> tuple[float, list[dict[str, object]], float]:
    model.eval()
    losses: list[float] = []
    rows: list[dict[str, object]] = []
    inference_times_ms: list[float] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            mask_labels = batch["mask_labels"].to(device)

            start = time.perf_counter()
            logits = model(images)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            inference_times_ms.append(elapsed_ms / max(int(images.size(0)), 1))

            loss = segmentation_loss(
                logits,
                masks,
                mask_labels,
                ignore_background=ignore_background_in_dice,
            )
            losses.append(float(loss.item()))

            pred_labels = torch.argmax(torch.softmax(logits, dim=1), dim=1).cpu().numpy()
            true_labels = mask_labels.cpu().numpy()

            batch_size = pred_labels.shape[0]
            for index in range(batch_size):
                sample_metrics = multiclass_overlap_metrics(
                    true_labels[index],
                    pred_labels[index],
                    class_values=class_values,
                    ignore_background=True,
                )
                rows.append(
                    {
                        "patient_id": batch["patient_id"][index],
                        "mode": mode,
                        "phase": batch["phase"][index],
                        "dataset": batch["dataset"][index],
                        "subset": batch["subset"][index],
                        **sample_metrics,
                    }
                )

    mean_loss = float(np.mean(losses)) if losses else float("nan")
    mean_inference_ms = float(np.mean(inference_times_ms)) if inference_times_ms else float("nan")
    return mean_loss, rows, mean_inference_ms


def summarize_metric_columns(df: pd.DataFrame) -> dict[str, float]:
    metric_cols = [
        col
        for col in df.columns
        if col.endswith("_dice") or col.endswith("_iou") or col.endswith("_hd95")
    ]
    summary: dict[str, float] = {}
    for column in metric_cols:
        series = df[column]
        if column.endswith("_hd95"):
            series = series.replace([np.inf, -np.inf], np.nan)
        summary[column] = float(series.mean()) if not series.empty else float("nan")
    return summary


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
    parser.add_argument("--mode", default="none")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--dataset-filter", nargs="*", default=[])
    parser.add_argument("--class-values", nargs="*", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-epochs", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    args = parser.parse_args()

    config = TrainConfig(
        preprocess_mode=args.mode,
        image_height=args.image_height,
        image_width=args.image_width,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        dataset_filter=tuple(args.dataset_filter),
        class_values=tuple(args.class_values),
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_epochs=args.early_stopping_min_epochs,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )

    if len(config.class_names) != len(config.class_values):
        raise ValueError("class_names and class_values must have identical lengths")

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

    export_split_manifests(split_map, output_dir / "resolved_splits", prefix="unet_multiclass")

    train_loader = build_dataloader(split_map["train"], config=config, augment=True, shuffle=True)
    val_loader = build_dataloader(split_map["val"], config=config, augment=False, shuffle=False)
    test_loader = build_dataloader(split_map["test"], config=config, augment=False, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet2D(out_channels=len(config.class_values)).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(config.epochs, 2))

    log(f"Starting multiclass run: mode={config.preprocess_mode}, output_dir={output_dir}")
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
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            ignore_background_in_dice=config.ignore_background_in_dice,
        )
        val_loss, _, val_inference_ms = evaluate_model(
            model,
            val_loader,
            device=device,
            mode=config.preprocess_mode,
            class_values=config.class_values,
            ignore_background_in_dice=config.ignore_background_in_dice,
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
        mode=config.preprocess_mode,
        class_values=config.class_values,
        ignore_background_in_dice=config.ignore_background_in_dice,
    )

    test_slice_df = pd.DataFrame(test_rows)
    test_slice_df.to_csv(output_dir / "test_slice_level.csv", index=False)
    patient_df = aggregate_patient_level(test_slice_df)
    patient_df.to_csv(output_dir / "test_patient_level.csv", index=False)

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
        "class_values": list(config.class_values),
        "class_names": list(config.class_names),
        "test_metrics_slice_mean": summarize_metric_columns(test_slice_df),
        "test_metrics_patient_mean": summarize_metric_columns(patient_df),
    }
    save_json(output_dir / "summary.json", summary)
    log(
        "Test summary: "
        f"loss={test_loss:.6f}, "
        f"macro_dice={summary['test_metrics_slice_mean'].get('macro_dice', float('nan')):.6f}, "
        f"macro_iou={summary['test_metrics_slice_mean'].get('macro_iou', float('nan')):.6f}, "
        f"macro_hd95={summary['test_metrics_slice_mean'].get('macro_hd95', float('nan')):.6f}, "
        f"inference_ms={test_inference_ms:.2f}"
    )

    save_runtime_log(
        output_dir,
        {
            "experiment": "train_unet_multiclass",
            "mode": config.preprocess_mode,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "device": str(device),
            "parameter_count": count_trainable_parameters(model),
            "manifest": str(args.manifest),
            "output_dir": str(output_dir),
            "class_values": ",".join(str(x) for x in config.class_values),
            "completed_epochs": completed_epochs,
            "best_epoch": best_epoch,
            "stopped_early": stopped_early,
        },
    )
    log("Run completed successfully.")


if __name__ == "__main__":
    main()
