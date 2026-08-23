from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from .data import CardiacImageDataset
from .metrics import binary_metrics
from .model import create_model, freeze_backbone, unfreeze_for_finetuning
from .transforms import CardiacTransform


def set_seed(seed: int) -> None:
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(frame, config, training: bool):
    from torch.utils.data import DataLoader
    dataset = CardiacImageDataset(frame, CardiacTransform(config.image_size, augment=training))
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=training,
                      num_workers=config.num_workers, pin_memory=True)


def _epoch(model, loader, device, optimizer=None):
    import torch
    criterion = torch.nn.BCEWithLogitsLoss()
    training = optimizer is not None
    model.train(training)
    losses, labels, scores, rows = [], [], [], []
    for images, targets, metadata in loader:
        images, targets = images.to(device), targets.float().to(device)
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images).flatten()
            loss = criterion(logits, targets)
            if training:
                loss.backward(); optimizer.step()
        probability = torch.sigmoid(logits).detach().cpu().numpy()
        losses.append(float(loss.item())); labels.extend(targets.cpu().numpy()); scores.extend(probability)
        for idx, score in enumerate(probability):
            rows.append({key: metadata[key][idx] for key in metadata} | {"label": int(targets[idx].item()), "probability": float(score)})
    return float(np.mean(losses)), binary_metrics(labels, scores), rows


def _fit_stage(model, train_loader, validation_loader, device, learning_rate, epochs, patience, checkpoint):
    import torch
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    best_auc, stale = -1.0, 0
    for _ in range(epochs):
        _epoch(model, train_loader, device, optimizer)
        _, metrics, _ = _epoch(model, validation_loader, device)
        score = metrics["roc_auc"]
        if np.isnan(score): score = metrics["accuracy"]
        if score > best_auc:
            best_auc, stale = score, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= patience: break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))


def train_cross_validation(frame: pd.DataFrame, config, output_dir: str | Path) -> pd.DataFrame:
    import torch
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for fold in sorted(frame["fold"].unique()):
        set_seed(config.seed + int(fold))
        train_frame, val_frame = frame[frame.fold != fold], frame[frame.fold == fold]
        train_loader, val_loader = _loader(train_frame, config, True), _loader(val_frame, config, False)
        model = create_model(config.architecture, config.dropout, config.pretrained).to(device)
        checkpoint = output / f"fold_{fold}.pt"
        freeze_backbone(model, config.architecture)
        _fit_stage(model, train_loader, val_loader, device, config.learning_rate_frozen,
                   config.epochs_frozen, config.early_stopping_patience, checkpoint)
        unfreeze_for_finetuning(model, config.architecture, config.vgg_blocks_to_unfreeze)
        _fit_stage(model, train_loader, val_loader, device, config.learning_rate_finetune,
                   config.epochs_finetune, config.early_stopping_patience, checkpoint)
        loss, metrics, predictions = _epoch(model, val_loader, device)
        metrics |= {"fold": int(fold), "loss": loss}
        results.append(metrics)
        pd.DataFrame(predictions).to_csv(output / f"fold_{fold}_predictions.csv", index=False)
    table = pd.DataFrame(results)
    table.to_csv(output / "cross_validation_metrics.csv", index=False)
    summary = table.select_dtypes(include="number").agg(["mean", "std"]).to_dict()
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return table


def evaluate_checkpoint(frame, config, checkpoint: str | Path, output: str | Path):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(config.architecture, config.dropout, pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    loss, metrics, predictions = _epoch(model, _loader(frame, config, False), device)
    metrics["loss"] = loss
    destination = Path(output); destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_csv(destination / "external_predictions.csv", index=False)
    (destination / "external_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
