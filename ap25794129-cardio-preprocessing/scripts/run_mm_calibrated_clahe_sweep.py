# -*- coding: utf-8 -*-
"""Runs the primary-budget (10-epoch, seed 42) comparison of none vs clahe vs
mm-calibrated clahe (clahe_mm) across ACDC, CAMUS, and the combined corpus.

clahe_kernel_size_mm is anchored at ACDC's mean native in-plane spacing
(1.5303 mm/px) so that the ACDC pixel-space kernel size is unchanged
(16 px) and CAMUS/combined get the physically equivalent kernel in their
own native pixel units.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
CLAHE_KERNEL_MM = 24.4848

MANIFESTS = {
    "acdc": ROOT / "data/manifests_local/acdc_all.csv",
    "camus": ROOT / "data/manifests_local/camus_all.csv",
    "combined": ROOT / "data/manifests_local/segmentation_public_combined.csv",
}

MODES = {
    "none": [],
    "clahe": ["--mode", "clahe"],
    "clahe_mm": ["--mode", "clahe", "--clahe-kernel-size-mm", str(CLAHE_KERNEL_MM)],
}

CACHE_DIR = ROOT / "outputs/_preprocess_cache_mmsweep"
OUT_ROOT = ROOT / "outputs/mm_calibrated_clahe_sweep"


def run_job(group: str, mode: str) -> None:
    out_dir = OUT_ROOT / f"{group}_{mode}"
    if (out_dir / "summary.json").exists():
        print(f"[SKIP] {group}/{mode} already has summary.json", flush=True)
        return
    cmd = [
        PY, "-m", "cardiac_image_system.experiments.train_unet_baseline",
        "--manifest", str(MANIFESTS[group]),
        "--output-dir", str(out_dir),
        "--epochs", "10",
        "--batch-size", "8",
        "--seed", "42",
        "--validation-seed", "42",
        "--preprocess-cache-dir", str(CACHE_DIR),
        *MODES[mode],
    ]
    print(f"\n=== RUN {group}/{mode} ===", flush=True)
    print(" ".join(cmd), flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    dt = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f"=== {group}/{mode}: {status} in {dt/60:.1f} min ===", flush=True)
    if result.returncode != 0:
        raise SystemExit(f"Job {group}/{mode} failed with code {result.returncode}")


def main() -> None:
    order = [
        ("acdc", "none"), ("acdc", "clahe"), ("acdc", "clahe_mm"),
        ("camus", "none"), ("camus", "clahe"), ("camus", "clahe_mm"),
        ("combined", "none"), ("combined", "clahe"), ("combined", "clahe_mm"),
    ]
    t_start = time.time()
    for group, mode in order:
        run_job(group, mode)
    print(f"\nALL DONE in {(time.time()-t_start)/60:.1f} min total", flush=True)


if __name__ == "__main__":
    main()
