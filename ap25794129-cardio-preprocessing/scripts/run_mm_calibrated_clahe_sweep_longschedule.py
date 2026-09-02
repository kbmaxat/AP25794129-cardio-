# -*- coding: utf-8 -*-
"""Long-schedule (up to 50 epochs, early stopping) confirmation of the primary-budget
mm-calibrated CLAHE result: none vs clahe vs clahe_mm across ACDC, CAMUS, and the
combined corpus. Same protocol (patience=10, min_epochs=15, min_delta=0.0005, seed=42)
used throughout the rest of the benchmark for long-schedule confirmatory reruns.
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
OUT_ROOT = ROOT / "outputs/mm_calibrated_clahe_sweep_longschedule"


def run_job(group: str, mode: str) -> None:
    out_dir = OUT_ROOT / f"{group}_{mode}"
    if (out_dir / "summary.json").exists():
        print(f"[SKIP] {group}/{mode} already has summary.json", flush=True)
        return
    cmd = [
        PY, "-m", "cardiac_image_system.experiments.train_unet_baseline",
        "--manifest", str(MANIFESTS[group]),
        "--output-dir", str(out_dir),
        "--epochs", "50",
        "--early-stopping-patience", "10",
        "--early-stopping-min-epochs", "15",
        "--early-stopping-min-delta", "0.0005",
        "--batch-size", "8",
        "--seed", "42",
        "--validation-seed", "42",
        "--preprocess-cache-dir", str(CACHE_DIR),
        *MODES[mode],
    ]
    print(f"\n=== RUN {group}/{mode} (long-schedule) ===", flush=True)
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
