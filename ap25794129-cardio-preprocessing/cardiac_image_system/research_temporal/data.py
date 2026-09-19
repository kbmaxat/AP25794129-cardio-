from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_splits(root: Path) -> dict[str, list[str]]:
    splits = {}
    for name in ("training", "validation", "testing"):
        lines = (root / "database_split" / f"subgroup_{name}.txt").read_text().splitlines()
        ids = [s.strip() for s in lines if s.strip()]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate patient in {name}")
        splits[name] = ids
    names = list(splits)
    for i, name in enumerate(names):
        for other in names[i + 1:]:
            if set(splits[name]) & set(splits[other]):
                raise ValueError(f"Patient leakage: {name}/{other}")
    return splits


def select_patients(splits: dict, config: dict) -> dict[str, list[str]]:
    if config["eligible_partition"] != "training" or config["external_test_access"]:
        raise ValueError("This pilot runner only permits the official training partition")
    ids = np.array(sorted(splits["training"]))
    rng = np.random.default_rng(config["split_seed"])
    ids = rng.permutation(ids).tolist()
    n, m = config["train_patients"], config["dev_patients"]
    if n < 1 or m < 1 or n + m > len(ids):
        raise ValueError("Invalid pilot sizes")
    return {"train": ids[:n], "dev": ids[n:n + m]}


def neighbors(center: int, frames: int) -> list[int]:
    if frames < 3 or not 0 <= center < frames:
        raise ValueError("Need at least three frames and a valid phase index")
    nearest = sorted((i for i in range(frames) if i != center), key=lambda i: (abs(i - center), i))[:2]
    return sorted(nearest)


def warp_neighbor(neighbor: np.ndarray, forward: np.ndarray):
    h, w = neighbor.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx, my = gx + forward[..., 0], gy + forward[..., 1]
    valid = (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
    warped = cv2.remap(neighbor, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return warped, mx, my, valid


def align_neighbor(center: np.ndarray, neighbor: np.ndarray, config: dict):
    p = config["alignment"]
    args = (None, p["pyr_scale"], p["levels"], p["winsize"], p["iterations"], p["poly_n"], p["poly_sigma"], 0)
    c8 = np.rint(center * 255).astype(np.uint8)
    n8 = np.rint(neighbor * 255).astype(np.uint8)
    forward = cv2.calcOpticalFlowFarneback(c8, n8, *args)
    backward = cv2.calcOpticalFlowFarneback(n8, c8, *args)
    warped, mx, my, valid = warp_neighbor(neighbor, forward)
    back_at_forward = cv2.remap(backward, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    fb_error = np.sum((forward + back_at_forward) ** 2, axis=-1)
    intensity_error = (center - warped) ** 2
    confidence = valid * np.exp(
        -fb_error / (2 * p["fb_sigma_px"] ** 2)
        -intensity_error / (2 * p["photometric_sigma"] ** 2)
    )
    return warped.astype(np.float32), confidence.astype(np.float32)


def load_cases(root: Path, patients: list[str], config: dict):
    samples, records = [], []
    for patient in patients:
        folder = root / "database_nifti" / patient
        view = config["view"]
        info_path = folder / f"Info_{view}.cfg"
        info = dict(line.split(":", 1) for line in info_path.read_text().splitlines() if ":" in line)
        info = {k.strip(): v.strip() for k, v in info.items()}
        seq_path = folder / f"{patient}_{view}_half_sequence.nii.gz"
        seq_nii = nib.load(seq_path)
        sequence = np.asarray(seq_nii.dataobj, dtype=np.float32)
        if sequence.ndim != 3 or not np.isfinite(sequence).all():
            raise ValueError(f"Invalid sequence {patient}: {sequence.shape}")
        fps = float(info["FrameRate"])
        if fps <= 0:
            raise ValueError("FrameRate must be positive")
        file_hashes = {str(seq_path): sha256(seq_path), str(info_path): sha256(info_path)}
        for phase in config["phases"]:
            center_idx = int(info[phase]) - 1
            near = neighbors(center_idx, sequence.shape[2])
            image_path = folder / f"{patient}_{view}_{phase}.nii.gz"
            mask_path = folder / f"{patient}_{view}_{phase}_gt.nii.gz"
            image_nii, mask_nii = nib.load(image_path), nib.load(mask_path)
            image = np.asarray(image_nii.dataobj, dtype=np.float32).squeeze()
            mask = np.asarray(mask_nii.dataobj).squeeze()
            if mask.shape != image.shape or image.shape != sequence.shape[:2]:
                raise ValueError(f"Geometry mismatch {patient}/{phase}")
            if not np.allclose(image_nii.affine, mask_nii.affine):
                raise ValueError(f"Image/mask affine mismatch {patient}/{phase}")
            if not np.allclose(image, sequence[..., center_idx], atol=1e-5):
                raise ValueError(f"Phase index does not match standalone image {patient}/{phase}")
            if not np.isin(mask, [0, 1, 2, 3]).all():
                raise ValueError(f"Unknown mask labels {patient}/{phase}")
            target = (mask == config["target_label"]).astype(np.float32)
            if not target.any():
                raise ValueError(f"Empty target {patient}/{phase}")
            lo, hi = np.percentile(image, [1, 99])
            if hi <= lo:
                raise ValueError(f"Constant image {patient}/{phase}")
            size = config["image_size"]
            raw = []
            for index in [center_idx, *near]:
                normalized = np.clip((sequence[..., index] - lo) / (hi - lo), 0, 1)
                raw.append(cv2.resize(normalized, (size, size), interpolation=cv2.INTER_AREA))
            raw = np.stack(raw).astype(np.float32)
            aligned, confidence = zip(*(align_neighbor(raw[0], n, config) for n in raw[1:]))
            resized_mask = cv2.resize(target, (size, size), interpolation=cv2.INTER_NEAREST)[None]
            samples.append({
                "raw": raw, "aligned": np.stack(aligned), "confidence": np.stack(confidence),
                "time_offsets": np.array([(i - center_idx) / fps for i in near], dtype=np.float32),
                "mask": resized_mask.astype(np.float32), "patient": patient, "phase": phase,
            })
            for path in (image_path, mask_path):
                file_hashes[str(path)] = sha256(path)
            records.append({
                "patient": patient, "phase": phase, "center_index_zero_based": center_idx,
                "neighbors_zero_based": near, "fps": fps, "quality": info["ImageQuality"],
                "shape_native": list(image.shape), "normalization_percentiles": [float(lo), float(hi)],
                "mean_alignment_confidence": float(np.mean(confidence)), "files_sha256": dict(file_hashes),
            })
    return samples, records
