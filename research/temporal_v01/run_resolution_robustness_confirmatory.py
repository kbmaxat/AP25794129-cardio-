"""Confirmatory experiment: does simple training-time augmentation on 2x-degraded images
close the segmentation-quality gap seen at evaluation time, on a held-out dev set never
used for fitting or checkpoint selection?

Registered protocol: resolution_robustness_confirmatory_protocol_v01.md
No CAMUS validation/testing access. Only the official CAMUS training-partition 320 patients
(further split into fit/select) and the 80 dev patients already reserved by
runs/pilot_20260927T093824Z_f0cd02/patient_split.json are used.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import io
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PACKAGE = ROOT / "ap25794129-cardio-preprocessing"
sys.path.insert(0, str(PACKAGE))

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from cardiac_image_system.research_temporal.data import sha256
from cardiac_image_system.models.unet2d import UNet2D

sys.path.insert(0, str(HERE))
from run_resolution_robustness_screening import (  # noqa: E402
    boundary_distances,
    load_phase_image,
    make_variant,
)

CHECKPOINT_RUN = HERE / "runs" / "pilot_20260927T093824Z_f0cd02"
SUBSPLIT_SEED = 20260930
N_SELECT = 32
VIEW = "4CH"
PHASES = ("ED", "ES")
IMAGE_SIZE = 128
BASE_CHANNELS = 16
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.001
BOUNDARY_WEIGHT = 0.1
AUGMENT_PROBABILITY = 0.5
SEEDS = (2027, 2028, 2029, 2030, 2031, 2032)
ARMS = ("clean_baseline", "degradation_augmented")
DEV_VARIANTS = ("clean", "2x_plain", "4x_plain")
SWEEP_EVAL_SIZE = 32
SWEEP_EVAL_SEED = 20261005


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def state_hash(model) -> str:
    buffer = io.BytesIO()
    for name, tensor in model.state_dict().items():
        buffer.write(name.encode())
        buffer.write(tensor.detach().cpu().numpy().tobytes())
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def loss_terms(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    pred = logits.sigmoid()
    dims = (1, 2, 3)
    soft_dice = ((2 * (pred * target).sum(dims) + 1e-6) / (pred.sum(dims) + target.sum(dims) + 1e-6)).mean()
    boundary = F.max_pool2d(target, 3, 1, 1) + F.max_pool2d(-target, 3, 1, 1)
    boundary_bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * boundary).sum() / boundary.sum().clamp_min(1)
    return bce + 1 - soft_dice + BOUNDARY_WEIGHT * boundary_bce


def build_subsplit(train_patients: list[str]) -> dict[str, list[str]]:
    ids = sorted(train_patients)
    rng = np.random.default_rng(SUBSPLIT_SEED)
    permuted = rng.permutation(np.array(ids)).tolist()
    return {"select": permuted[:N_SELECT], "fit": permuted[N_SELECT:]}


def build_sweep_subsplit(train_patients: list[str]) -> dict[str, list[str]]:
    initial = build_subsplit(train_patients)
    eval_rng = np.random.default_rng(SWEEP_EVAL_SEED)
    eval_permuted = eval_rng.permutation(np.array(sorted(initial["fit"]))).tolist()
    sweep_eval = eval_permuted[:SWEEP_EVAL_SIZE]
    fit = eval_permuted[SWEEP_EVAL_SIZE:]
    return {"select": initial["select"], "fit": fit, "sweep_evaluation": sweep_eval}


def load_patient_samples(data_root: Path, patients: list[str]) -> list[dict]:
    samples = []
    for patient in patients:
        for phase in PHASES:
            normalized, native_target = load_phase_image(data_root, patient, phase)
            clean_128 = cv2.resize(normalized, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
            mask_128 = cv2.resize(native_target, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)[None].astype(np.float32)
            samples.append({
                "patient": patient, "phase": phase, "clean_128": clean_128,
                "mask_128": mask_128, "native_target": native_target,
            })
        print(f"loaded patient={patient}", flush=True)
    return samples


class FitDataset(Dataset):
    """Training dataset. For the augmented arm, redraws per-sample degradation flags
    every epoch via set_epoch(); flags are deterministic given (training_seed, epoch)."""

    def __init__(
        self,
        samples: list[dict],
        augment: bool,
        training_seed: int,
        augmentation_variant: str,
        augmentation_probability: float,
    ):
        self.samples = samples
        self.augment = augment
        self.training_seed = training_seed
        self.augmentation_variant = augmentation_variant
        self.augmentation_probability = augmentation_probability
        self.flags = np.zeros(len(samples), dtype=bool)

    def set_epoch(self, epoch: int) -> None:
        if self.augment:
            rng = np.random.default_rng((self.training_seed, epoch))
            self.flags = rng.random(len(self.samples)) < self.augmentation_probability
        else:
            self.flags[:] = False

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image = sample["clean_128"]
        if self.flags[index]:
            image = make_variant(image, self.augmentation_variant)
        return {
            "image": torch.from_numpy(image)[None, :, :],
            "mask": torch.from_numpy(sample["mask_128"]),
        }


def fit_collate(batch: list[dict]) -> dict:
    return {
        "image": torch.utils.data.default_collate([item["image"] for item in batch]),
        "mask": torch.utils.data.default_collate([item["mask"] for item in batch]),
    }


def evaluate_loss(model, samples: list[dict], device: str) -> float:
    """Loss on the select set, always on clean images, identical criterion for both arms."""
    model.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, len(samples), BATCH_SIZE):
            chunk = samples[start:start + BATCH_SIZE]
            images = torch.from_numpy(np.stack([s["clean_128"] for s in chunk]))[:, None, :, :].to(device)
            masks = torch.from_numpy(np.stack([s["mask_128"] for s in chunk])).to(device)
            logits = model(images)
            loss = loss_terms(logits, masks)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite select loss")
            losses.append(float(loss))
    return float(np.mean(losses))


def evaluate_dev(model, samples: list[dict], variant: str, device: str) -> list[dict]:
    model.eval()
    rows = []
    with torch.no_grad():
        for sample in samples:
            image = sample["clean_128"] if variant == "clean" else make_variant(sample["clean_128"], variant)
            tensor = torch.from_numpy(image)[None, None, :, :].to(device)
            logits = model(tensor)
            probability = logits[0, 0].sigmoid().cpu().numpy()
            native_target = sample["native_target"]
            native_size = (native_target.shape[1], native_target.shape[0])
            native_probability = cv2.resize(probability, native_size, interpolation=cv2.INTER_LINEAR)
            native_pred = native_probability >= 0.5
            native_mask = native_target.astype(bool)
            intersection = np.logical_and(native_pred, native_mask).sum()
            dice = (2 * intersection + 1e-6) / (native_pred.sum() + native_mask.sum() + 1e-6)
            area_error = abs(native_pred.sum() - native_mask.sum()) / max(native_mask.sum(), 1)
            mean_surface_distance, hausdorff_distance = boundary_distances(native_pred, native_mask)
            rows.append({
                "patient": sample["patient"], "phase": sample["phase"], "variant": variant,
                "dice": float(dice), "relative_area_error": float(area_error),
                "mean_surface_distance_px": mean_surface_distance,
                "hausdorff_distance_px": hausdorff_distance,
            })
    return rows


def train_arm(
    arm: str,
    seed: int,
    fit_samples,
    select_samples,
    dev_samples,
    output_dir: Path,
    device: str,
    augmentation_variant: str,
    augmentation_probability: float,
) -> dict:
    arm_dir = output_dir / f"seed{seed}_{arm}"
    arm_dir.mkdir(parents=True)
    seed_everything(seed)
    model = UNet2D(in_channels=1, base_channels=BASE_CHANNELS)
    initial_hash = state_hash(model)
    model.to(device)
    dataset = FitDataset(
        fit_samples,
        augment=(arm == "degradation_augmented"),
        training_seed=seed,
        augmentation_variant=augmentation_variant,
        augmentation_probability=augmentation_probability,
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator, num_workers=0, collate_fn=fit_collate)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_loss, best_epoch = float("inf"), None
    history = []
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        dataset.set_epoch(epoch)
        model.train()
        epoch_losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            loss = loss_terms(logits, masks)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Nonfinite gradients")
            optimizer.step()
            epoch_losses.extend([float(loss.detach())] * len(images))
        select_loss = evaluate_loss(model, select_samples, device)
        if select_loss < best_loss:
            best_loss, best_epoch = select_loss, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "arm": arm, "seed": seed}, arm_dir / "best.pt")
        row = {"epoch": epoch, "train_loss": float(np.mean(epoch_losses)), "select_loss": select_loss}
        history.append(row)
        write_csv(arm_dir / "history.csv", history)
        print(f"seed={seed} arm={arm} epoch={epoch}/{EPOCHS} train={row['train_loss']:.4f} select={select_loss:.4f}", flush=True)
    seconds = time.perf_counter() - started
    checkpoint = torch.load(arm_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    all_rows = []
    for variant in DEV_VARIANTS:
        all_rows.extend(evaluate_dev(model, dev_samples, variant, device))
    write_csv(arm_dir / "dev_phase_metrics.csv", all_rows)
    summary = {
        "seed": seed, "arm": arm, "best_epoch": best_epoch, "best_select_loss": best_loss,
        "training_seconds": seconds, "initial_segmenter_sha256": initial_hash,
        "checkpoint_sha256": sha256(arm_dir / "best.pt"),
        "parameters": sum(p.numel() for p in model.parameters()),
        "augmentation_variant": augmentation_variant if arm == "degradation_augmented" else "none",
        "augmentation_probability": augmentation_probability if arm == "degradation_augmented" else 0.0,
    }
    write_json(arm_dir / "summary.json", summary)
    del model, optimizer
    if device == "cuda":
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--augmentation-variant", choices=("2x_plain", "4x_plain"), default="2x_plain")
    parser.add_argument("--augmentation-probability", type=float, default=AUGMENT_PROBABILITY)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=HERE / "resolution_robustness_confirmatory_protocol_v01.md",
    )
    parser.add_argument(
        "--exploratory-sweep",
        action="store_true",
        help="Use an internal training-partition sweep holdout; never access official dev",
    )
    args = parser.parse_args()

    if not 0.0 <= args.augmentation_probability <= 1.0:
        parser.error("--augmentation-probability must be in [0, 1]")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required; this confirmatory run is not silently run on CPU")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)

    patient_split = json.loads((CHECKPOINT_RUN / "patient_split.json").read_text(encoding="utf-8"))
    train_patients = sorted(patient_split["train"])
    dev_patients = sorted(patient_split["dev"])
    if len(train_patients) != 320 or len(dev_patients) != 80:
        raise ValueError("Unexpected patient counts in reused split")
    if args.exploratory_sweep:
        subsplit = build_sweep_subsplit(train_patients)
        if (
            len(subsplit["select"]) != N_SELECT
            or len(subsplit["sweep_evaluation"]) != SWEEP_EVAL_SIZE
            or len(subsplit["fit"]) != 320 - N_SELECT - SWEEP_EVAL_SIZE
        ):
            raise ValueError("Sweep subsplit sizes do not match protocol")
        partitions = [set(subsplit[name]) for name in ("fit", "select", "sweep_evaluation")]
        if any(partitions[i] & partitions[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("fit/select/sweep_evaluation overlap")
        if set.union(*partitions) != set(train_patients):
            raise ValueError("fit/select/sweep_evaluation do not partition train patients")
    else:
        subsplit = build_subsplit(train_patients)
        if len(subsplit["select"]) != N_SELECT or len(subsplit["fit"]) != 320 - N_SELECT:
            raise ValueError("Subsplit sizes do not match protocol")
        if set(subsplit["select"]) & set(subsplit["fit"]):
            raise ValueError("fit/select overlap")
        if set(subsplit["select"]) | set(subsplit["fit"]) != set(train_patients):
            raise ValueError("fit/select does not partition train patients")

    private_output = args.private_output.resolve()
    private_output.mkdir(parents=True, exist_ok=False)
    write_json(private_output / "patient_subsplit.json", subsplit)

    data_root = args.data_root.resolve()
    print("Loading fit patients...", flush=True)
    fit_samples = load_patient_samples(data_root, subsplit["fit"])
    print("Loading select patients...", flush=True)
    select_samples = load_patient_samples(data_root, subsplit["select"])
    if args.exploratory_sweep:
        print("Loading sweep evaluation patients (official dev is not accessed)...", flush=True)
        dev_samples = load_patient_samples(data_root, subsplit["sweep_evaluation"])
    else:
        print("Loading official dev patients...", flush=True)
        dev_samples = load_patient_samples(data_root, dev_patients)

    all_summaries = []
    for seed in args.seeds:
        for arm in ARMS:
            summary = train_arm(
                arm, seed, fit_samples, select_samples, dev_samples, private_output, args.device,
                args.augmentation_variant, args.augmentation_probability,
            )
            all_summaries.append(summary)
            write_csv(private_output / "arm_summaries.csv", all_summaries)

    metadata = {
        "experiment_id": (
            "RESOLUTION-AUGMENTATION-PARAMETER-SCREEN-001"
            if args.exploratory_sweep else "RESOLUTION-ROBUSTNESS-CONFIRMATORY-001"
        ),
        "stage": (
            "exploratory_train_partition_augmentation_screening"
            if args.exploratory_sweep else "confirmatory"
        ),
        "utc_finished": now(),
        "protocol_sha256": sha256(args.protocol.resolve()),
        "protocol_file": args.protocol.name,
        "reused_split_run": CHECKPOINT_RUN.name,
        "n_fit_patients": len(subsplit["fit"]),
        "n_select_patients": len(subsplit["select"]),
        "seeds": list(args.seeds),
        "arms": list(ARMS),
        "dev_variants": list(DEV_VARIANTS),
        "augmentation_variant": args.augmentation_variant,
        "augmentation_probability": args.augmentation_probability,
        "test_access": False,
    }
    if args.exploratory_sweep:
        metadata["n_sweep_evaluation_patients"] = len(subsplit["sweep_evaluation"])
        metadata["official_dev_accessed"] = False
        metadata["evaluation_partition"] = "training_partition_sweep_holdout"
    else:
        metadata["n_dev_patients"] = len(dev_patients)
        metadata["evaluation_partition"] = "official_dev"
    write_json(private_output / "metadata.json", metadata)
    print(f"OUTPUT={private_output}")


if __name__ == "__main__":
    main()
