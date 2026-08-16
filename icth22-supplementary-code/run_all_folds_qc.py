from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent
if (SCRIPT_ROOT.parent / "src").is_dir():
    PROJECT_ROOT = SCRIPT_ROOT.parent
    DATASET_ROOT = PROJECT_ROOT / "data" / "HeartDatabase"
    OUTPUT = PROJECT_ROOT / "results" / "all_folds_qc_reproduced.json"
else:
    ARTICLE_ROOT = SCRIPT_ROOT.parent
    PROJECT_ROOT = ARTICLE_ROOT.parent / "cardiac_image_system"
    DATASET_ROOT = (
        PROJECT_ROOT
        / "data"
        / "data"
        / "raw"
        / "HeartDatabase"
        / "HeartDatabase"
    )
    OUTPUT = ARTICLE_ROOT / "ICTH2026_assets" / "all_folds_qc.json"

sys.path.insert(0, str(PROJECT_ROOT))

from src.augmentation import augment_pair, sample_parameters, validate_pair  # noqa: E402
from src.heart_database import discover_samples, load_sample  # noqa: E402
from src.prepare_heart_dataset import (  # noqa: E402
    build_folds,
    split_for_fold,
    stable_seed,
)


GLOBAL_SEED = 25794129
COPIES_PER_TRAINING_VOLUME = 4


def sample_stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def evaluate_folds() -> list[dict]:
    samples = discover_samples(DATASET_ROOT)
    loaded = {sample.sample_id: load_sample(sample) for sample in samples}
    patient_ids = sorted({sample.patient_id for sample in samples})
    folds = build_folds(patient_ids, 5, GLOBAL_SEED)
    results: list[dict] = []

    for fold_index in range(5):
        split = split_for_fold(folds, fold_index)
        training_patients = set(split["train"])
        cavity_ratios: list[float] = []
        myocardium_ratios: list[float] = []
        started = time.perf_counter()

        for sample in samples:
            if sample.patient_id not in training_patients:
                continue
            image, label, _ = loaded[sample.sample_id]
            for copy_index in range(1, COPIES_PER_TRAINING_VOLUME + 1):
                item_seed = stable_seed(
                    GLOBAL_SEED,
                    sample.patient_id,
                    sample.phase,
                    str(copy_index),
                )
                rng = np.random.default_rng(item_seed)
                parameters = sample_parameters(rng)
                augmented_image, augmented_label = augment_pair(
                    image, label, parameters, rng
                )
                qc = validate_pair(augmented_image, augmented_label, label)
                cavity_ratios.append(float(qc["cavity_volume_ratio"]))
                myocardium_ratios.append(float(qc["myocardium_volume_ratio"]))

        results.append(
            {
                "fold": fold_index,
                "train_patients": len(split["train"]),
                "validation_patients": len(split["validation"]),
                "test_patients": len(split["test"]),
                "augmented_pairs": len(cavity_ratios),
                "qc_passed": len(cavity_ratios),
                "cavity_volume_ratio": sample_stats(cavity_ratios),
                "myocardium_volume_ratio": sample_stats(myocardium_ratios),
                "augmentation_and_qc_seconds": time.perf_counter() - started,
            }
        )
    return results


def expect_rejection(
    image: np.ndarray,
    label: np.ndarray,
    reference: np.ndarray,
) -> bool:
    try:
        validate_pair(image, label, reference)
    except ValueError:
        return True
    return False


def challenge_cases(
    image: np.ndarray,
    label: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    nonfinite = image.copy()
    nonfinite.flat[0] = np.nan

    out_of_range = image.copy()
    out_of_range.flat[0] = 1.01

    illegal_label = label.copy()
    illegal_label.flat[0] = 3

    missing_class = label.copy()
    missing_class[missing_class == 1] = 0

    volume_loss = label.copy()
    cavity_indices = np.flatnonzero(volume_loss == 1)
    volume_loss.flat[cavity_indices[: int(np.ceil(0.21 * len(cavity_indices)))]] = 0

    return {
        "shape_mismatch": (image[:-1], label),
        "nonfinite_intensity": (nonfinite, label),
        "out_of_range_intensity": (out_of_range, label),
        "illegal_label": (image, illegal_label),
        "missing_foreground_class": (image, missing_class),
        "foreground_volume_loss": (image, volume_loss),
    }


def evaluate_qc_challenge_suite() -> dict:
    samples = discover_samples(DATASET_ROOT)
    detected_by_type: dict[str, int] = {}
    total_by_type: dict[str, int] = {}

    for sample in samples:
        image, label, _ = load_sample(sample)
        for name, (challenged_image, challenged_label) in challenge_cases(
            image, label
        ).items():
            total_by_type[name] = total_by_type.get(name, 0) + 1
            detected_by_type[name] = detected_by_type.get(name, 0) + int(
                expect_rejection(challenged_image, challenged_label, label)
            )

    total = sum(total_by_type.values())
    detected = sum(detected_by_type.values())
    return {
        "failure_modes": len(total_by_type),
        "samples_per_failure_mode": len(samples),
        "total_injected_failures": total,
        "detected_failures": detected,
        "detection_rate": detected / total,
        "detected_by_type": detected_by_type,
        "total_by_type": total_by_type,
        "interpretation": (
            "A deterministic gate-verification suite; not a substitute for "
            "clinical or anatomical validation."
        ),
    }


def main() -> None:
    output = {
        "global_seed": GLOBAL_SEED,
        "copies_per_training_volume": COPIES_PER_TRAINING_VOLUME,
        "folds": evaluate_folds(),
        "qc_challenge_suite": evaluate_qc_challenge_suite(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
