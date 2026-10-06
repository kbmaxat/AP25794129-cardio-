"""Summarize fixed-candidate EchoNet-Dynamic external evaluation.

Video IDs are the patient-level bootstrap unit. Outputs contain aggregate metrics only.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PACKAGE = ROOT / "ap25794129-cardio-preprocessing"
sys.path.insert(0, str(PACKAGE))

import numpy as np

from cardiac_image_system.research_temporal.data import sha256

SEEDS = (2027, 2028, 2029, 2030, 2031, 2032)
CONFIGS = {
    "p025_2x": "2x_plain",
    "p075_2x": "2x_plain",
    "p050_4x": "4x_plain",
}
METRICS = ("dice", "relative_area_error", "mean_surface_distance_px", "hausdorff_distance_px")
N_BOOTSTRAP = 10000


def read_patient_values(path: Path) -> dict[tuple[int, str, str, str], dict[str, float]]:
    grouped: dict[tuple[int, str, str, str], list[dict[str, str]]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["seed"]), row["arm"], row["condition"], row["video_id"])
            grouped.setdefault(key, []).append(row)
    if not grouped:
        raise ValueError("No EchoNet external evaluation metrics")
    result = {}
    for key, frames in grouped.items():
        if len(frames) != 2 or len({row["frame"] for row in frames}) != 2:
            raise ValueError(f"Expected exactly two distinct traced frames for {key}")
        result[key] = {
            metric: sum(float(row[metric]) for row in frames) / 2 for metric in METRICS
        }
    return result


def crossed_bootstrap(values: np.ndarray, seed: int) -> list[float]:
    if values.ndim != 2 or values.shape[0] != len(SEEDS) or values.shape[1] < 2:
        raise ValueError(f"Expected 6xN crossed values, got {values.shape}")
    rng = np.random.default_rng(seed)
    estimates = np.empty(N_BOOTSTRAP, dtype=np.float64)
    for index in range(N_BOOTSTRAP):
        selected_seeds = rng.integers(0, values.shape[0], values.shape[0])
        selected_videos = rng.integers(0, values.shape[1], values.shape[1])
        estimates[index] = values[np.ix_(selected_seeds, selected_videos)].mean()
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path, required=True)
    args = parser.parse_args()
    private_output = args.private_output.resolve()
    metadata = json.loads((private_output / "metadata.json").read_text(encoding="utf-8"))
    if (
        metadata["experiment_id"] != "RESOLUTION-AUGMENTATION-ECHONET-EXTERNAL-001"
        or metadata["stage"] != "external_evaluation_fixed_candidates"
        or metadata["dataset_split"] != "VAL"
        or metadata["n_val_videos_listed"] != 1288
        or metadata["n_videos"] != 1173
        or metadata["n_val_videos_excluded_undecodable"] != 115
        or metadata["n_annotated_frames"] != metadata["n_videos"] * 2
        or metadata["official_CAMUS_validation_or_testing_accessed"] is not False
        or metadata["EchoNet_train_or_test_videos_opened"] is not False
        or metadata["seeds"] != list(SEEDS)
    ):
        raise ValueError("Run metadata does not match the registered external evaluation")

    per_video = read_patient_values(private_output / "echonet_val_frame_metrics.csv")
    video_ids = sorted({key[3] for key in per_video})
    if len(video_ids) != metadata["n_videos"]:
        raise ValueError(f"Video unit count differs from metadata: {len(video_ids)}")
    expected_keys = set()
    for seed in SEEDS:
        for condition in ("clean", "2x_plain", "4x_plain"):
            expected_keys.add((seed, "clean_baseline", condition))
        for config, degraded in CONFIGS.items():
            arm = f"aug_{config}"
            expected_keys.add((seed, arm, "clean"))
            expected_keys.add((seed, arm, degraded))
    observed = {(seed, arm, condition) for seed, arm, condition, _ in per_video}
    if observed != expected_keys:
        raise ValueError("Observed arm/condition combinations differ from protocol")
    for seed, arm, condition in observed:
        videos = {video for s, a, c, video in per_video if (s, a, c) == (seed, arm, condition)}
        if videos != set(video_ids):
            raise ValueError(f"Video coverage mismatch for seed={seed}, arm={arm}, {condition}")

    def matrix(arm: str, condition: str, metric: str) -> np.ndarray:
        values = np.empty((len(SEEDS), len(video_ids)), dtype=np.float64)
        for si, seed in enumerate(SEEDS):
            for vi, video in enumerate(video_ids):
                values[si, vi] = per_video[(seed, arm, condition, video)][metric]
        return values

    result = {
        "experiment_id": metadata["experiment_id"],
        "stage": metadata["stage"],
        "status_note": (
            "External evaluation of fixed candidates on EchoNet-Dynamic VAL. All candidates "
            "are reported; the cohort must not be reused to select one and claim a new confirmation."
        ),
        "dataset": metadata["dataset_name"],
        "split": metadata["dataset_split"],
        "n_videos": len(video_ids),
        "n_videos_listed_in_split": metadata["n_val_videos_listed"],
        "n_videos_excluded_undecodable": metadata["n_val_videos_excluded_undecodable"],
        "n_annotated_frames": metadata["n_annotated_frames"],
        "n_seeds": len(SEEDS),
        "bootstrap_replicates": N_BOOTSTRAP,
        "official_CAMUS_validation_or_testing_accessed": False,
        "EchoNet_train_or_test_videos_opened": False,
        "candidate_configs": {},
        "protocol_sha256": metadata["protocol_sha256"],
        "interpretation_note": (
            "CAMUS-trained segmenters had near-zero absolute EchoNet VAL Dice; relative "
            "augmentation contrasts are floor-limited and do not demonstrate useful "
            "cross-dataset segmentation or resolution robustness."
        ),
    }
    for config, degraded in CONFIGS.items():
        arm = f"aug_{config}"
        result["candidate_configs"][config] = {
            "degraded_condition": degraded,
            "means": {},
            "contrasts": {},
        }
        for (arm_key, condition), label in (
            (("clean_baseline", "clean"), "clean_baseline_clean"),
            (("clean_baseline", degraded), "clean_baseline_degraded"),
            ((arm, "clean"), "augmentation_clean"),
            ((arm, degraded), "augmentation_degraded"),
        ):
            result["candidate_configs"][config]["means"][label] = {
                metric: float(matrix(arm_key, condition, metric).mean()) for metric in METRICS
            }
        contrasts = {
            "baseline_degradation_degraded_minus_clean": (
                ("clean_baseline", degraded), ("clean_baseline", "clean"),
            ),
            "augmentation_minus_baseline_on_degraded": (
                (arm, degraded), ("clean_baseline", degraded),
            ),
            "clean_guardrail_augmentation_minus_baseline": (
                (arm, "clean"), ("clean_baseline", "clean"),
            ),
            "residual_gap_augmentation_degraded_minus_baseline_clean": (
                (arm, degraded), ("clean_baseline", "clean"),
            ),
        }
        for ci, (contrast_name, ((arm_a, condition_a), (arm_b, condition_b))) in enumerate(contrasts.items()):
            result["candidate_configs"][config]["contrasts"][contrast_name] = {}
            for mi, metric in enumerate(METRICS):
                delta = matrix(arm_a, condition_a, metric) - matrix(arm_b, condition_b, metric)
                result["candidate_configs"][config]["contrasts"][contrast_name][metric] = {
                    "mean_delta": float(delta.mean()),
                    "crossed_video_seed_bootstrap_95_ci": crossed_bootstrap(
                        delta, seed=20261060 + list(CONFIGS).index(config) * 100 + ci * 10 + mi
                    ),
                }

    output = args.aggregate_output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary_file = output / "resolution_augmentation_echonet_external_aggregate.json"
    summary_file.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "experiment_id": metadata["experiment_id"],
        "stage": metadata["stage"],
        "protocol_sha256": metadata["protocol_sha256"],
        "private_metadata_sha256": sha256(private_output / "metadata.json"),
        "private_data_audit_sha256": sha256(private_output / "data_audit.json"),
        "private_exclusion_manifest_sha256": sha256(private_output / "excluded_val_videos.csv"),
        "private_frame_metrics_sha256": sha256(private_output / "echonet_val_frame_metrics.csv"),
        "checkpoint_hashes": metadata["checkpoint_hashes"],
        "echo_file_list_sha256": metadata["source_hashes"]["file_list"],
        "echo_volume_tracings_sha256": metadata["source_hashes"]["volume_tracings"],
        "video_ids_included": False,
        "official_CAMUS_validation_or_testing_accessed": False,
        "EchoNet_train_or_test_videos_opened": False,
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    hashes = {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()}
    (output / "files_sha256.json").write_text(
        json.dumps(hashes, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=True, indent=2))
    print(f"OUTPUT={output}")


if __name__ == "__main__":
    main()
