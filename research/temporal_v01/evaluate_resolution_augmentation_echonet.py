"""External EchoNet-Dynamic VAL evaluation for fixed CAMUS-trained augmentation candidates.

No training, checkpoint selection, or CAMUS validation/testing access occurs here.
EchoNet TRAIN and TEST videos are never opened.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PACKAGE = ROOT / "ap25794129-cardio-preprocessing"
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(HERE))

import cv2
import numpy as np
import torch
from skimage.draw import polygon

from cardiac_image_system.models.unet2d import UNet2D
from cardiac_image_system.research_temporal.data import sha256
from run_resolution_robustness_screening import boundary_distances, make_variant

SEEDS = (2027, 2028, 2029, 2030, 2031, 2032)
CONFIGS = {
    "p025_2x": {"dir": "resolution_augmentation_screen_p025_2x_20261005", "variant": "2x_plain", "p": 0.25},
    "p075_2x": {"dir": "resolution_augmentation_screen_p075_2x_20261005", "variant": "2x_plain", "p": 0.75},
    "p050_4x": {"dir": "resolution_augmentation_screen_p050_4x_20261005", "variant": "4x_plain", "p": 0.50},
}
BASELINE_KEY = "clean_baseline"
IMAGE_SIZE = 128
EXPECTED_VAL_VIDEOS = 1288
METRICS = ("dice", "relative_area_error", "mean_surface_distance_px", "hausdorff_distance_px")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def write_csv_header(path: Path, fieldnames: list[str]):
    stream = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    return stream, writer


def state_hash(model) -> str:
    import hashlib
    import io

    buffer = io.BytesIO()
    for name, tensor in model.state_dict().items():
        buffer.write(name.encode())
        buffer.write(tensor.detach().cpu().numpy().tobytes())
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def read_validation_ids(filelist_path: Path) -> tuple[list[dict[str, str]], set[str]]:
    with filelist_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    val_rows = [row for row in rows if row["Split"].strip().upper() == "VAL"]
    if len(val_rows) != EXPECTED_VAL_VIDEOS:
        raise ValueError(f"Expected {EXPECTED_VAL_VIDEOS} EchoNet VAL videos, found {len(val_rows)}")
    ids = {row["FileName"].removesuffix(".avi") for row in val_rows}
    if len(ids) != len(val_rows):
        raise ValueError("Duplicate EchoNet VAL video IDs")
    return val_rows, ids


def read_val_traces(trace_path: Path, val_filenames: set[str]) -> dict[str, dict[int, np.ndarray]]:
    traces: dict[str, dict[int, list[tuple[float, float, float, float]]]] = {}
    with trace_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        expected = ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]
        if reader.fieldnames != expected:
            raise ValueError(f"Unexpected VolumeTracings header: {reader.fieldnames}")
        for row in reader:
            filename = row["FileName"]
            if filename not in val_filenames:
                continue
            values = tuple(float(row[key]) for key in ("X1", "Y1", "X2", "Y2"))
            frame = int(row["Frame"])
            traces.setdefault(filename, {}).setdefault(frame, []).append(values)
    if set(traces) != val_filenames:
        raise ValueError(
            f"Missing VAL traces for {len(val_filenames - set(traces))} videos"
        )
    if any(len(frames) != 2 for frames in traces.values()):
        raise ValueError("Every EchoNet VAL video must have exactly two traced frames")
    return {
        filename: {
            frame: np.asarray(segments, dtype=np.float64)
            for frame, segments in frame_map.items()
        }
        for filename, frame_map in traces.items()
    }


def rasterize_trace(segments: np.ndarray, shape: tuple[int, int], video: str, frame: int) -> np.ndarray:
    if segments.ndim != 2 or segments.shape[1] != 4 or len(segments) < 3:
        raise ValueError(f"Invalid tracing rows for {video}/{frame}: {segments.shape}")
    if not np.isfinite(segments).all():
        raise ValueError(f"Nonfinite tracing coordinates for {video}/{frame}")
    height, width = shape

    x1, y1, x2, y2 = (segments[:, index] for index in range(4))
    x = np.concatenate((x1[1:], np.flip(x2[1:])))
    y = np.concatenate((y1[1:], np.flip(y2[1:])))
    rows, cols = polygon(np.rint(y).astype(np.intp), np.rint(x).astype(np.intp), shape)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[rows, cols] = 1
    if not mask.any():
        raise ValueError(f"Empty rasterized tracing for {video}/{frame}")
    return mask


def prepare_echo_frame(video_path: Path, frame_index: int, expected_shape: tuple[int, int]) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"Could not open EchoNet VAL video: {video_path.name}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise OSError(f"Could not decode EchoNet VAL frame {video_path.name}/{frame_index}")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.shape != expected_shape:
        raise ValueError(
            f"Frame size mismatch for {video_path.name}/{frame_index}: "
            f"{gray.shape} != {expected_shape}"
        )
    low, high = np.percentile(gray, [1, 99])
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError(f"Invalid intensity range for {video_path.name}/{frame_index}")
    normalized = np.clip((gray.astype(np.float32) - low) / (high - low), 0.0, 1.0)
    resized = cv2.resize(
        normalized, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA
    ).astype(np.float32)
    return resized


def preflight_val_videos(
    echonet_root: Path,
    val_rows: list[dict[str, str]],
    traces: dict[str, dict[int, np.ndarray]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    eligible = []
    excluded = []
    for row in val_rows:
        video_id = row["FileName"].removesuffix(".avi")
        filename = f"{video_id}.avi"
        video_path = echonet_root / "Videos" / filename
        if not video_path.is_file():
            excluded.append({"video_id": video_id, "reason": "missing_file"})
            continue
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            excluded.append({"video_id": video_id, "reason": "video_open_failed"})
            continue
        expected_shape = (int(row["FrameHeight"]), int(row["FrameWidth"]))
        failure = None
        for frame_index in sorted(traces[filename]):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                failure = f"annotated_frame_decode_failed:{frame_index}"
                break
            if frame.shape[:2] != expected_shape:
                failure = f"frame_dimension_mismatch:{frame_index}"
                break
        capture.release()
        if failure is None:
            eligible.append(row)
        else:
            excluded.append({"video_id": video_id, "reason": failure})
    return eligible, excluded


def load_echo_val(echonet_root: Path) -> tuple[list[dict], dict, list[dict[str, str]]]:
    videos_dir = echonet_root / "Videos"
    filelist_path = echonet_root / "FileList.csv"
    trace_path = echonet_root / "VolumeTracings.csv"
    val_rows, val_ids = read_validation_ids(filelist_path)
    val_filenames = {f"{video_id}.avi" for video_id in val_ids}
    traces = read_val_traces(trace_path, val_filenames)
    eligible_rows, excluded = preflight_val_videos(echonet_root, val_rows, traces)
    if len(eligible_rows) != 1173 or len(excluded) != 115:
        raise ValueError(
            f"Decoder preflight changed: eligible={len(eligible_rows)}, excluded={len(excluded)}; "
            "review protocol amendment before any model inference"
        )
    records = []
    coordinate_min = np.full(4, np.inf, dtype=np.float64)
    coordinate_max = np.full(4, -np.inf, dtype=np.float64)
    out_of_frame_coordinate_count = 0
    total_coordinate_count = 0
    for row in eligible_rows:
        video_id = row["FileName"].removesuffix(".avi")
        filename = f"{video_id}.avi"
        expected_shape = (int(row["FrameHeight"]), int(row["FrameWidth"]))
        if expected_shape != (112, 112):
            raise ValueError(f"Unexpected EchoNet frame dimensions for {video_id}: {expected_shape}")
        video_path = videos_dir / filename
        if not video_path.is_file():
            raise FileNotFoundError(f"Missing EchoNet VAL video: {filename}")
        for frame_index, segments in sorted(traces[filename].items()):
            coordinate_min = np.minimum(coordinate_min, segments.min(axis=0))
            coordinate_max = np.maximum(coordinate_max, segments.max(axis=0))
            height, width = expected_shape
            out_of_frame_coordinate_count += int(
                (segments[:, (0, 2)] < 0).sum()
                + (segments[:, (0, 2)] > width - 1).sum()
                + (segments[:, (1, 3)] < 0).sum()
                + (segments[:, (1, 3)] > height - 1).sum()
            )
            total_coordinate_count += int(segments.size)
            native_mask = rasterize_trace(segments, expected_shape, filename, frame_index)
            clean_128 = prepare_echo_frame(video_path, frame_index, expected_shape)
            records.append({
                "video_id": video_id,
                "frame": frame_index,
                "clean_128": clean_128,
                "native_mask": native_mask,
            })
        print(f"prepared EchoNet VAL video={video_id}", flush=True)
    audit = {
        "n_val_videos_listed": len(val_rows),
        "n_val_videos_eligible": len(eligible_rows),
        "n_val_videos_excluded_undecodable": len(excluded),
        "n_val_annotated_frames": len(records),
        "native_shape": [112, 112],
        "image_size": IMAGE_SIZE,
        "mask_area_min": int(min(record["native_mask"].sum() for record in records)),
        "mask_area_max": int(max(record["native_mask"].sum() for record in records)),
        "mask_area_mean": float(np.mean([record["native_mask"].sum() for record in records])),
        "out_of_frame_coordinate_values": out_of_frame_coordinate_count,
        "total_coordinate_values": total_coordinate_count,
        "coordinate_min_x1_y1_x2_y2": coordinate_min.tolist(),
        "coordinate_max_x1_y1_x2_y2": coordinate_max.tolist(),
        "train_or_test_video_frames_opened": False,
    }
    return records, audit, excluded


def evaluate_condition(
    model: UNet2D,
    records: list[dict],
    condition: str,
    device: str,
    seed: int,
    arm: str,
    writer: csv.DictWriter,
    batch_size: int = 64,
) -> None:
    model.eval()
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        images = []
        for record in chunk:
            clean = record["clean_128"]
            images.append(
                clean if condition == "clean" else make_variant(clean, condition)
            )
        tensor = torch.from_numpy(np.stack(images))[:, None, :, :].to(device)
        with torch.no_grad():
            probabilities = model(tensor).sigmoid()[:, 0].cpu().numpy()
        for record, probability in zip(chunk, probabilities):
            height, width = record["native_mask"].shape
            native_probability = cv2.resize(
                probability, (width, height), interpolation=cv2.INTER_LINEAR
            )
            prediction = native_probability >= 0.5
            target = record["native_mask"].astype(bool)
            intersection = np.logical_and(prediction, target).sum()
            dice = (2 * intersection + 1e-6) / (prediction.sum() + target.sum() + 1e-6)
            area_error = abs(prediction.sum() - target.sum()) / max(target.sum(), 1)
            mean_surface_distance, hausdorff_distance = boundary_distances(prediction, target)
            writer.writerow({
                "seed": seed,
                "arm": arm,
                "condition": condition,
                "video_id": record["video_id"],
                "frame": record["frame"],
                "dice": float(dice),
                "relative_area_error": float(area_error),
                "mean_surface_distance_px": mean_surface_distance,
                "hausdorff_distance_px": hausdorff_distance,
            })
        print(
            f"evaluated seed={seed} arm={arm} condition={condition} "
            f"frames={min(start + batch_size, len(records))}/{len(records)}",
            flush=True,
        )


def load_model(checkpoint_path: Path, expected_hash: str, device: str) -> tuple[UNet2D, str]:
    actual_hash = sha256(checkpoint_path)
    if actual_hash != expected_hash:
        raise ValueError(f"Checkpoint hash mismatch at {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = UNet2D(in_channels=1, base_channels=16)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().to(device)
    return model, actual_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--echonet-root", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument(
        "--screen-root",
        type=Path,
        default=HERE / "private",
        help="Parent directory containing the three completed candidate screen runs",
    )
    parser.add_argument("--protocol", type=Path, default=HERE / "resolution_augmentation_echonet_external_protocol_v01.md")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required; external evaluation will not silently run on CPU")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)

    screen_root = args.screen_root.resolve()
    run_dirs = {
        name: screen_root / spec["dir"] for name, spec in CONFIGS.items()
    }
    baseline_hashes = {}
    candidate_hashes = {}
    subsplit_hashes = set()
    for name, run_dir in run_dirs.items():
        meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        if (
            meta.get("stage") != "exploratory_train_partition_augmentation_screening"
            or meta.get("official_dev_accessed") is not False
            or meta.get("test_access") is not False
            or meta.get("seeds") != list(SEEDS)
            or meta.get("augmentation_variant") != CONFIGS[name]["variant"]
            or meta.get("augmentation_probability") != CONFIGS[name]["p"]
        ):
            raise ValueError(f"Unexpected parameter-screen metadata for {name}")
        subsplit_hashes.add(sha256(run_dir / "patient_subsplit.json"))
        summaries_path = run_dir / "arm_summaries.csv"
        with summaries_path.open(encoding="utf-8", newline="") as stream:
            summaries = list(csv.DictReader(stream))
        for seed in SEEDS:
            for arm, target in (
                (BASELINE_KEY, baseline_hashes),
                ("degradation_augmented", candidate_hashes),
            ):
                rows = [r for r in summaries if int(r["seed"]) == seed and r["arm"] == arm]
                if len(rows) != 1:
                    raise ValueError(f"Expected one {arm} summary for seed {seed}, config {name}")
                if arm == BASELINE_KEY:
                    if seed in target and target[seed] != rows[0]["checkpoint_sha256"]:
                        raise ValueError(f"Baseline checkpoint differs across candidate runs for seed {seed}")
                    target[seed] = rows[0]["checkpoint_sha256"]
                else:
                    target[(name, seed)] = rows[0]["checkpoint_sha256"]
    if len(subsplit_hashes) != 1:
        raise ValueError("Candidate runs do not share a common internal patient split")

    output = args.private_output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records, data_audit, excluded_videos = load_echo_val(args.echonet_root.resolve())
    write_json(output / "data_audit.json", data_audit)
    exclusion_path = output / "excluded_val_videos.csv"
    with exclusion_path.open("w", newline="", encoding="utf-8") as exclusion_stream:
        exclusion_writer = csv.DictWriter(exclusion_stream, fieldnames=["video_id", "reason"])
        exclusion_writer.writeheader()
        exclusion_writer.writerows(excluded_videos)

    metric_path = output / "echonet_val_frame_metrics.csv"
    fields = [
        "seed", "arm", "condition", "video_id", "frame", *METRICS,
    ]
    stream, writer = write_csv_header(metric_path, fields)
    loaded_hashes = {}
    try:
        for seed in SEEDS:
            baseline_path = run_dirs["p025_2x"] / f"seed{seed}_{BASELINE_KEY}" / "best.pt"
            baseline_model, baseline_hash = load_model(
                baseline_path, baseline_hashes[seed], args.device
            )
            loaded_hashes[f"baseline_seed{seed}"] = baseline_hash
            for condition in ("clean", "2x_plain", "4x_plain"):
                evaluate_condition(
                    baseline_model, records, condition, args.device, seed,
                    BASELINE_KEY, writer,
                )
            del baseline_model
            if args.device == "cuda":
                torch.cuda.empty_cache()

            for name, spec in CONFIGS.items():
                arm = f"aug_{name}"
                checkpoint_path = run_dirs[name] / f"seed{seed}_degradation_augmented" / "best.pt"
                model, checkpoint_hash = load_model(
                    checkpoint_path, candidate_hashes[(name, seed)], args.device
                )
                loaded_hashes[f"{name}_seed{seed}"] = checkpoint_hash
                for condition in ("clean", spec["variant"]):
                    evaluate_condition(
                        model, records, condition, args.device, seed, arm, writer,
                    )
                del model
                if args.device == "cuda":
                    torch.cuda.empty_cache()
    finally:
        stream.close()

    metadata = {
        "experiment_id": "RESOLUTION-AUGMENTATION-ECHONET-EXTERNAL-001",
        "stage": "external_evaluation_fixed_candidates",
        "utc_finished": now(),
        "protocol_sha256": sha256(args.protocol.resolve()),
        "protocol_file": args.protocol.name,
        "dataset_name": "EchoNet-Dynamic",
        "dataset_split": "VAL",
        "n_videos": data_audit["n_val_videos_eligible"],
        "n_val_videos_listed": data_audit["n_val_videos_listed"],
        "n_val_videos_excluded_undecodable": data_audit["n_val_videos_excluded_undecodable"],
        "n_annotated_frames": data_audit["n_val_annotated_frames"],
        "seeds": list(SEEDS),
        "candidate_configs": CONFIGS,
        "patient_unit": "video_id",
        "bootstrap_replicates": 10000,
        "official_CAMUS_validation_or_testing_accessed": False,
        "EchoNet_train_or_test_videos_opened": False,
        "checkpoint_hashes": loaded_hashes,
        "candidate_screen_split_sha256": next(iter(subsplit_hashes)),
        "source_hashes": {
            "runner": sha256(Path(__file__).resolve()),
            "screen_runner": sha256(HERE / "run_resolution_robustness_confirmatory.py"),
            "screen_helper": sha256(HERE / "run_resolution_robustness_screening.py"),
            "file_list": sha256(args.echonet_root.resolve() / "FileList.csv"),
            "volume_tracings": sha256(args.echonet_root.resolve() / "VolumeTracings.csv"),
        },
        "scikit_image_version": __import__("skimage").__version__,
        "opencv_version": cv2.__version__,
        "pytorch_version": torch.__version__,
        "data_audit": data_audit,
    }
    write_json(output / "metadata.json", metadata)
    print(f"OUTPUT={output}")


if __name__ == "__main__":
    main()
