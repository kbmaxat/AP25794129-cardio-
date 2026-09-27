from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "ap25794129-cardio-preprocessing"))

import cv2
import nibabel as nib
import numpy as np

from cardiac_image_system.research_temporal.data import read_splits, select_patients
from cardiac_image_system.research_temporal.geometry import compute_letterbox_transform, apply_letterbox
from export_pilot import csv_rows, read_json, sha256, write_csv, write_json


def roundtrip(root, patients, size):
    rows = []
    for patient in patients:
        for phase in ("ED", "ES"):
            image_path = root / "database_nifti" / patient / f"{patient}_4CH_{phase}_gt.nii.gz"
            nii = nib.load(image_path)
            target = (np.asarray(nii.dataobj).squeeze() == 1).astype(np.uint8)
            h, w = target.shape
            transform = compute_letterbox_transform(target.shape, size)
            for mode in ("squash", "letterbox"):
                if mode == "squash":
                    small = cv2.resize(target, (size, size), interpolation=cv2.INTER_NEAREST)
                    content = small
                else:
                    small = apply_letterbox(target, transform, cv2.INTER_NEAREST)
                    rh, rw = round(h * transform.scale), round(w * transform.scale)
                    top, left = transform.pad_top, transform.pad_left
                    content = small[top:top + rh, left:left + rw]
                restored = cv2.resize(content, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                true = target.astype(bool)
                dice = 2 * np.logical_and(restored, true).sum() / (restored.sum() + true.sum())
                rows.append({"patient": patient, "phase": phase, "geometry": mode,
                             "native_height": h, "native_width": w,
                             "spacing_axis0": float(nii.header.get_zooms()[0]),
                             "spacing_axis1": float(nii.header.get_zooms()[1]),
                             "dice": float(dice),
                             "area_error": abs(int(restored.sum()) - int(true.sum())) / int(true.sum()),
                             "mask_sha256": sha256(image_path)})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    plan = read_json(HERE / "geometry_check_plan.json")
    baseline = read_json(HERE / "protocol_v01.json")
    for key in ("train_patients", "dev_patients", "split_seed", "image_size"):
        assert baseline[key] == plan[key], key
    stamp = datetime.now(timezone.utc).strftime("geometry_check_%Y%m%dT%H%M%SZ")
    folder = HERE / "private" / stamp
    folder.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, folder / Path(__file__).name)
    write_json(folder / "plan.json", plan)
    write_json(folder / "status.json", {"status": "running", "utc": datetime.now(timezone.utc).isoformat()})
    selected = select_patients(read_splits(args.data_root), baseline)
    write_json(folder / "patient_split.json", selected)
    rt = roundtrip(args.data_root, selected["train"] + selected["dev"], plan["image_size"])
    write_csv(folder / "roundtrip_phase.csv", rt)
    aggregate_rt = []
    for mode in plan["geometries"]:
        values = [r for r in rt if r["geometry"] == mode]
        aggregate_rt.append({"geometry": mode, "n_patients": len(selected["train"] + selected["dev"]),
                             "dice": statistics.mean(r["dice"] for r in values),
                             "relative_area_error": statistics.mean(r["area_error"] for r in values)})
    print(json.dumps({"roundtrip": aggregate_rt}), flush=True)
    results, provenance = [], {}
    try:
        for seed in plan["seeds"]:
            initial_hash = None
            split_hash = None
            for geometry in plan["geometries"]:
                command = [sys.executable, str(HERE / "run_pilot.py"), "--data-root", str(args.data_root),
                           "--training-seed", str(seed), "--epochs", str(plan["epochs"]),
                           "--modes", "none", "--geometry-mode", geometry]
                env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
                completed = subprocess.run(command, cwd=ROOT, env=env, text=True, encoding="utf-8",
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                (folder / f"seed_{seed}_{geometry}.log").write_text(completed.stdout, encoding="utf-8")
                if completed.returncode:
                    raise RuntimeError(f"Run failed: {seed}/{geometry}; see saved log")
                directory = [line.split("=", 1)[1] for line in completed.stdout.splitlines() if line.startswith("RUN_DIRECTORY=")]
                if len(directory) != 1:
                    raise ValueError("Missing unique run directory")
                run = Path(directory[0])
                summary = read_json(run / "none/summary.json")
                conf = read_json(run / "protocol.json")
                assert conf["geometry_mode"] == geometry and conf["training_seed"] == seed
                current_split = sha256(run / "patient_split.json")
                if initial_hash is not None:
                    assert summary["initial_segmenter_sha256"] == initial_hash
                    assert current_split == split_hash
                initial_hash, split_hash = summary["initial_segmenter_sha256"], current_split
                phase_rows = csv_rows(run / "none/dev_phase_metrics.csv")
                metrics = ("dice", "relative_area_error", "mean_surface_distance_px", "hausdorff_distance_px")
                by_patient = defaultdict(list)
                for row in phase_rows:
                    by_patient[row["patient"]].append(row)
                assert len(by_patient) == plan["dev_patients"]
                means = {key: statistics.mean(statistics.mean(float(r[key]) for r in rs)
                                             for rs in by_patient.values()) for key in metrics}
                result = {"seed": seed, "geometry": geometry, "run_id": run.name,
                          "n_dev_patients": len(by_patient), "epochs": plan["epochs"],
                          "best_epoch": summary["best_epoch"], **means}
                results.append(result)
                provenance[run.name] = {"protocol_sha256": sha256(run / "protocol.json"),
                                        "phase_metrics_sha256": sha256(run / "none/dev_phase_metrics.csv"),
                                        "source_tree_sha256": read_json(run / "environment.json")["source_tree_sha256"],
                                        "initial_segmenter_sha256": initial_hash, "patient_split_sha256": split_hash}
                write_csv(folder / "results_progress.csv", results)
                print(json.dumps(result), flush=True)
        deltas = []
        for seed in plan["seeds"]:
            pair = {r["geometry"]: r for r in results if r["seed"] == seed}
            deltas.append({"seed": seed, **{key: pair["letterbox"][key] - pair["squash"][key] for key in metrics}})
        output = HERE / "public_results" / stamp
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "plan.json", plan)
        write_csv(output / "native_dev_metrics.csv", results)
        write_csv(output / "paired_seed_differences.csv", deltas)
        write_csv(output / "mask_roundtrip.csv", aggregate_rt)
        write_json(output / "aggregate.json", {
            "stage": "development_exploratory", "n_unique_dev_patients": plan["dev_patients"],
            "n_splits": 1, "n_seeds": len(plan["seeds"]), "test_access": False,
            "delta_definition": "letterbox_minus_squash", "p_values_computed": False,
            "metrics": {key: {"mean_delta": statistics.mean(r[key] for r in deltas),
                              "sd_across_seeds_ddof1": statistics.stdev(r[key] for r in deltas)} for key in metrics},
        })
        write_json(output / "provenance.json", provenance)
        write_json(output / "files_sha256.json", {p.name: sha256(p) for p in sorted(output.iterdir())})
        write_json(folder / "status.json", {"status": "complete", "public_directory": output.name})
        print(f"PUBLIC_RESULTS={output}", flush=True)
    except BaseException as error:
        write_json(folder / "status.json", {"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
