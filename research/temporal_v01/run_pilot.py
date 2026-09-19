from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback
import uuid

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PACKAGE = ROOT / "ap25794129-cardio-preprocessing"
sys.path.insert(0, str(PACKAGE))

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from cardiac_image_system.research_temporal.data import load_cases, read_splits, select_patients, sha256
from cardiac_image_system.research_temporal.model import TemporalSystem


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def event(kind, **fields):
    record = {"utc": now(), "kind": kind, **fields}
    with (HERE / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=True), flush=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def state_hash(model):
    buffer = io.BytesIO()
    # Hash values rather than serialized storage identifiers.
    for name, tensor in model.state_dict().items():
        buffer.write(name.encode())
        buffer.write(tensor.detach().cpu().numpy().tobytes())
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def loss_terms(logits, target, change, config):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    pred = logits.sigmoid()
    dims = (1, 2, 3)
    soft_dice = ((2 * (pred * target).sum(dims) + 1e-6) / (pred.sum(dims) + target.sum(dims) + 1e-6)).mean()
    boundary = F.max_pool2d(target, 3, 1, 1) + F.max_pool2d(-target, 3, 1, 1)
    boundary_bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * boundary).sum() / boundary.sum().clamp_min(1)
    correction = change.abs().mean()
    return bce + 1 - soft_dice + config["boundary_weight"] * boundary_bce + config["correction_weight"] * correction


def forward_batch(model, batch, device):
    inputs = [batch[key].to(device) for key in ("raw", "aligned", "confidence", "time_offsets")]
    return model(*inputs)


def evaluate(model, loader, config, device):
    model.eval()
    losses, rows = [], []
    with torch.no_grad():
        for batch in loader:
            target = batch["mask"].to(device)
            logits, image, change = forward_batch(model, batch, device)
            loss = loss_terms(logits, target, change, config)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite dev loss")
            losses.extend([float(loss)] * len(target))
            pred = (logits.sigmoid() >= 0.5).float()
            dims = (1, 2, 3)
            dice = (2 * (pred * target).sum(dims) + 1e-6) / (pred.sum(dims) + target.sum(dims) + 1e-6)
            area_error = (pred.sum(dims) - target.sum(dims)).abs() / target.sum(dims).clamp_min(1)
            for i in range(len(target)):
                rows.append({
                    "patient": batch["patient"][i], "phase": batch["phase"][i],
                    "dice": float(dice[i]), "relative_area_error": float(area_error[i]),
                    "mean_abs_correction": float(change[i].abs().mean()),
                    "max_abs_correction": float(change[i].abs().max()),
                })
    by_patient = defaultdict(list)
    for row in rows:
        by_patient[row["patient"]].append(row["dice"])
    patient_dice = {key: float(np.mean(value)) for key, value in by_patient.items()}
    return float(np.mean(losses)), rows, patient_dice


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_provenance(run, config_path):
    sources = [p for p in (PACKAGE / "cardiac_image_system/research_temporal").glob("*.py")]
    sources += [PACKAGE / "cardiac_image_system/models/unet2d.py", Path(__file__), config_path]
    sources += list(HERE.glob("*.md"))
    sources += list((PACKAGE / "tests").glob("test_research_temporal.py"))
    hashes = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        destination = run / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[str(relative)] = sha256(source)
    write_json(run / "source_hashes.json", hashes)
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (run / "requirements_freeze.txt").write_text(freeze, encoding="utf-8")
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    git_status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    (run / "git_status.txt").write_text(git_status, encoding="utf-8")
    env = {
        "utc": now(), "python": sys.version, "executable": sys.executable,
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(), "numpy": np.__version__, "opencv": cv2.__version__,
        "base_commit": git_head, "protocol_sha256": sha256(config_path),
        "source_tree_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
    }
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
        env["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    write_json(run / "environment.json", env)
    return env


def train_mode(mode, train, dev, config, run, baseline_state, device):
    output = run / mode
    output.mkdir()
    seed_everything(config["training_seed"])
    model = TemporalSystem(mode, config)
    state = model.segmenter.state_dict()
    for key, tensor in baseline_state.items():
        if tensor.shape == state[key].shape:
            state[key] = tensor.clone()
        elif key == "inc.block.0.weight" and mode == "direct_temporal":
            state[key] = tensor.repeat(1, 3, 1, 1) / 3
        else:
            raise ValueError(f"Unexpected initial state mismatch: {key}")
    model.segmenter.load_state_dict(state)
    initial_hash = state_hash(model.segmenter)
    model.to(device)
    generator = torch.Generator().manual_seed(config["training_seed"])
    train_loader = DataLoader(train, batch_size=config["batch_size"], shuffle=True, generator=generator, num_workers=0)
    dev_loader = DataLoader(dev, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    best, best_epoch = float("inf"), None
    history = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    event("mode_started", run_id=run.name, mode=mode, initial_segmenter_sha256=initial_hash)
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            target = batch["mask"].to(device)
            logits, _, change = forward_batch(model, batch, device)
            loss = loss_terms(logits, target, change, config)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Nonfinite gradients")
            optimizer.step()
            epoch_losses.extend([float(loss.detach())] * len(target))
        dev_loss, _, patient_dice = evaluate(model, dev_loader, config, device)
        if dev_loss < best:
            best, best_epoch = dev_loss, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": config, "mode": mode}, output / "best.pt")
        row = {"epoch": epoch, "train_loss": float(np.mean(epoch_losses)), "dev_loss": dev_loss, "dev_patient_dice": float(np.mean(list(patient_dice.values())))}
        history.append(row)
        write_csv(output / "history.csv", history)
        print(f"{mode} epoch={epoch}/{config['epochs']} train={row['train_loss']:.4f} dev={dev_loss:.4f} Dice_dev={row['dev_patient_dice']:.4f}", flush=True)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated()
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    dev_loss, rows, patients = evaluate(model, dev_loader, config, device)
    write_csv(output / "dev_phase_metrics.csv", rows)
    write_csv(output / "dev_patient_metrics.csv", [{"patient": p, "dice": d} for p, d in patients.items()])
    sample = next(iter(DataLoader(dev[:1], batch_size=1)))
    with torch.no_grad():
        logits, processed, change = forward_batch(model, sample, device)
    np.savez_compressed(output / "fixed_dev_example.npz", raw=sample["raw"][0].numpy(), aligned=sample["aligned"][0].numpy(), confidence=sample["confidence"][0].numpy(), mask=sample["mask"][0, 0].numpy(), processed=processed[0, 0].cpu().numpy(), correction=change[0, 0].cpu().numpy(), prediction=(logits[0, 0].sigmoid() >= 0.5).cpu().numpy())
    summary = {
        "mode": mode, "stage": "technical_pilot_not_confirmatory", "best_epoch": best_epoch,
        "dev_loss": dev_loss, "dev_patient_dice": float(np.mean(list(patients.values()))),
        "dev_n_patients": len(patients), "training_seconds": seconds,
        "gpu_peak_allocated_bytes": peak, "parameters": sum(p.numel() for p in model.parameters()),
        "preprocessor_parameters": sum(p.numel() for p in model.preprocessor.parameters()) if model.preprocessor else 0,
        "mean_abs_correction": float(np.mean([row["mean_abs_correction"] for row in rows])),
        "max_abs_correction": max(row["max_abs_correction"] for row in rows),
        "initial_segmenter_sha256": initial_hash, "checkpoint_sha256": sha256(output / "best.pt"),
    }
    if summary["max_abs_correction"] > config["epsilon"] + 1e-6:
        raise ValueError("Correction bound violated")
    write_json(output / "summary.json", summary)
    event("mode_completed", run_id=run.name, **summary)
    del model, optimizer
    torch.cuda.empty_cache()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "protocol_v01.json")
    parser.add_argument("--data-root", type=Path, default=os.environ.get("CAMUS_ROOT"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.data_root is not None:
        config["dataset_root"] = str(args.data_root.resolve())
    if not config["dataset_root"]:
        parser.error("Provide --data-root or CAMUS_ROOT")
    if config["stage"] != "technical_pilot" or config["bootstrap_or_hypothesis_testing"]:
        raise ValueError("This runner does not implement confirmatory analysis")
    run_id = datetime.now(timezone.utc).strftime("pilot_%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:6]
    run = HERE / "runs" / run_id
    run.mkdir(parents=True)
    write_json(run / "protocol.json", config)
    event("run_started", run_id=run_id, protocol_sha256=sha256(args.config), stage=config["stage"])
    started = time.perf_counter()
    try:
        env = save_provenance(run, args.config)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; this pilot is not silently run on CPU")
        torch.set_num_threads(config["torch_threads"])
        cv2.setNumThreads(1)
        root = Path(config["dataset_root"])
        splits = read_splits(root)
        selected = select_patients(splits, config)
        write_json(run / "patient_split.json", selected)
        data, records = {}, {}
        for partition, patients in selected.items():
            data[partition], records[partition] = load_cases(root, patients, config)
            print(f"Prepared {partition}: {len(patients)} patients, {len(data[partition])} phases", flush=True)
        write_json(run / "data_manifest.json", records)
        event("data_ready", run_id=run_id, train_patients=len(selected["train"]), dev_patients=len(selected["dev"]), access="official_CAMUS_training_only")
        seed_everything(config["training_seed"])
        baseline = TemporalSystem("none", config)
        baseline_state = {key: value.clone() for key, value in baseline.segmenter.state_dict().items()}
        del baseline
        summaries = [train_mode(mode, data["train"], data["dev"], config, run, baseline_state, "cuda") for mode in config["modes"]]
        write_csv(run / "summary.csv", summaries)
        write_json(run / "status.json", {"status": "completed", "utc": now(), "seconds": time.perf_counter() - started, "claim": "technical feasibility only", "environment": env})
        with (HERE / "ЖУРНАЛ_ИССЛЕДОВАНИЯ.md").open("a", encoding="utf-8") as journal:
            journal.write(f"\n## {now()} Завершен технический пилот {run_id}\n\n")
            journal.write("CAMUS training only; n_train=32, n_dev=8; checkpoint selection: dev loss.\n\n")
            for item in summaries:
                journal.write(f"- {item['mode']}: Dice dev {item['dev_patient_dice']:.4f}; эпоха {item['best_epoch']}; обучение {item['training_seconds']:.1f} с; максимальная поправка {item['max_abs_correction']:.5f}.\n")
            journal.write(f"\nАртефакты: runs/{run_id}. Конфигурация, данные, код и окружение зафиксированы контрольными суммами.\n")
        event("run_completed", run_id=run_id, seconds=time.perf_counter() - started)
        print(f"RUN_DIRECTORY={run}", flush=True)
    except BaseException as error:
        trace = traceback.format_exc()
        (run / "error.txt").write_text(trace, encoding="utf-8")
        write_json(run / "status.json", {"status": "failed", "utc": now(), "error": repr(error)})
        event("run_failed", run_id=run_id, error=repr(error))
        with (HERE / "ЖУРНАЛ_ИССЛЕДОВАНИЯ.md").open("a", encoding="utf-8") as journal:
            journal.write(f"\n## {now()} Неудачный запуск {run_id}\n\nОшибка: `{error!r}`. Исходники, конфигурация и traceback сохранены в runs/{run_id}.\n")
        raise


if __name__ == "__main__":
    main()
