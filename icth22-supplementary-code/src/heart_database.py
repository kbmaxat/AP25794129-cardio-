"""Reader and sample index for the open 3D+t cardiac MRI Heart Database."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SYSTOLE_PATTERN = re.compile(r"Systole\s*=\s*(out\d+\.pgm)", re.IGNORECASE)


@dataclass(frozen=True)
class HeartSample:
    patient_id: str
    phase: str
    image_path: Path
    expert1_endo_path: Path
    expert1_epi_path: Path
    expert2_endo_path: Path
    expert2_epi_path: Path

    @property
    def sample_id(self) -> str:
        return f"{self.patient_id}_{self.phase}"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_p7_volume(path: Path) -> np.ndarray:
    """Read the database's P7 extension of PGM as a (z, y, x) uint8 array."""
    with path.open("rb") as stream:
        if stream.readline().strip() != b"P7":
            raise ValueError(f"{path}: expected P7 magic")

        tokens: list[bytes] = []
        while len(tokens) < 4:
            line = stream.readline()
            if not line:
                raise ValueError(f"{path}: incomplete P7 header")
            tokens.extend(line.split(b"#", 1)[0].split())

        width, height, depth, max_value = map(int, tokens[:4])
        if max_value > 255:
            raise ValueError(f"{path}: only 8-bit P7 volumes are supported")
        payload = stream.read()

    expected_size = width * height * depth
    if len(payload) != expected_size:
        raise ValueError(
            f"{path}: payload has {len(payload)} bytes, expected {expected_size}"
        )
    return np.frombuffer(payload, dtype=np.uint8).reshape(depth, height, width)


def _phase_image_names(patient_dir: Path) -> dict[str, str]:
    info = (patient_dir / "info.txt").read_text(encoding="ascii")
    match = SYSTOLE_PATTERN.search(info)
    if not match:
        raise ValueError(f"{patient_dir}: systolic frame is missing from info.txt")
    return {"diastole": "out001.pgm", "systole": match.group(1)}


def discover_samples(dataset_root: Path) -> list[HeartSample]:
    samples: list[HeartSample] = []
    patient_dirs = sorted(path for path in dataset_root.glob("Pat*") if path.is_dir())
    if not patient_dirs:
        raise ValueError(f"No PatXX directories found in {dataset_root}")

    for patient_dir in patient_dirs:
        for phase, image_name in _phase_image_names(patient_dir).items():
            sample = HeartSample(
                patient_id=patient_dir.name,
                phase=phase,
                image_path=patient_dir / "img" / image_name,
                expert1_endo_path=patient_dir
                / "expert1"
                / f"{phase}_endocarde_scaled.pgm",
                expert1_epi_path=patient_dir
                / "expert1"
                / f"{phase}_epicarde_scaled.pgm",
                expert2_endo_path=patient_dir
                / "expert2"
                / f"{phase}_endocarde_scaled.pgm",
                expert2_epi_path=patient_dir
                / "expert2"
                / f"{phase}_epicarde_scaled.pgm",
            )
            missing = [
                path
                for path in (
                    sample.image_path,
                    sample.expert1_endo_path,
                    sample.expert1_epi_path,
                    sample.expert2_endo_path,
                    sample.expert2_epi_path,
                )
                if not path.is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"{sample.sample_id}: missing {', '.join(map(str, missing))}"
                )
            samples.append(sample)
    return samples


def build_label(endocardium: np.ndarray, epicardium: np.ndarray) -> np.ndarray:
    """Build labels: 0 background, 1 LV cavity, 2 myocardium."""
    if endocardium.shape != epicardium.shape:
        raise ValueError("Endocardial and epicardial masks have different shapes")
    cavity = endocardium > 0
    outer = epicardium > 0
    if np.any(cavity & ~outer):
        raise ValueError("Endocardial mask is not contained in epicardial mask")

    label = np.zeros(endocardium.shape, dtype=np.uint8)
    label[outer & ~cavity] = 2
    label[cavity] = 1
    return label


def robust_normalize(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    foreground = image[image > 0]
    if foreground.size == 0:
        raise ValueError("Image contains no positive voxels")
    low, high = np.percentile(foreground, (1.0, 99.0))
    if high <= low:
        raise ValueError("Image intensity range is degenerate")
    return np.clip((image - low) / (high - low), 0.0, 1.0).astype(np.float32)


def load_sample(sample: HeartSample) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = read_p7_volume(sample.image_path)
    expert1 = build_label(
        read_p7_volume(sample.expert1_endo_path),
        read_p7_volume(sample.expert1_epi_path),
    )
    expert2 = build_label(
        read_p7_volume(sample.expert2_endo_path),
        read_p7_volume(sample.expert2_epi_path),
    )
    if image.shape != expert1.shape or image.shape != expert2.shape:
        raise ValueError(f"{sample.sample_id}: image and masks have different shapes")
    return robust_normalize(image), expert1, expert2
