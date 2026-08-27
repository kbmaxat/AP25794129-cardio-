# -*- coding: utf-8 -*-
"""Primary-budget (10-epoch, seed 42) comparison of mm-calibrated nlm (nlm_mm) and
mm-calibrated hybrid (hybrid_mm) against the existing none/clahe/clahe_mm runs already
in outputs/mm_calibrated_clahe_sweep/, across ACDC, CAMUS, and the combined corpus.

mm targets are anchored at ACDC's mean native in-plane spacing (1.5303 mm/px), same
anchor as the CLAHE calibration, so ACDC's own pixel-space parameters are unchanged and
CAMUS/combined get the physically equivalent kernel/patch sizes in their native pixels.

Order (camus, combined, acdc) is deliberate: camus populates the preprocessing disk
cache for the (CPU-bound) NLM step, and combined shares CAMUS's image files, so running
camus first lets combined reuse those cached entries instead of recomputing NLM on the
same ~2000 CAMUS images a second time.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

CLAHE_KERNEL_MM = 24.4848
NLM_PATCH_SIZE_MM = 7.6515
NLM_PATCH_DISTANCE_MM = 9.1818

MANIFESTS = {
    "acdc": ROOT / "data/manifests_local/acdc_all.csv",
    "camus": ROOT / "data/manifests_local/camus_all.csv",
    "combined": ROOT / "data/manifests_local/segmentation_public_combined.csv",
}

MM_FLAGS = [
    "--nlm-patch-size-mm", str(NLM_PATCH_SIZE_MM),
    "--nlm-patch-distance-mm", str(NLM_PATCH_DISTANCE_MM),
]

MODES = {
    "nlm_mm": ["--mode", "nlm", *MM_FLAGS],
    "hybrid_mm": ["--mode", "hybrid", *MM_FLAGS, "--clahe-kernel-size-mm", str(CLAHE_KERNEL_MM)],
}

CACHE_DIR = ROOT / "outputs/_preprocess_cache_mmsweep"
OUT_ROOT = ROOT / "outputs/mm_calibrated_clahe_sweep"  # same output root as the CLAHE sweep


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
        ("camus", "nlm_mm"), ("camus", "hybrid_mm"),
        ("combined", "nlm_mm"), ("combined", "hybrid_mm"),
        ("acdc", "nlm_mm"), ("acdc", "hybrid_mm"),
    ]
    t_start = time.time()
    for group, mode in order:
        run_job(group, mode)
    print(f"\nALL DONE in {(time.time()-t_start)/60:.1f} min total", flush=True)


if __name__ == "__main__":
    main()
