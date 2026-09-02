# -*- coding: utf-8 -*-
"""Primary-budget (10-epoch, seed 42) AdaptiveGateUNet run across ACDC, CAMUS, the
combined corpus, and ImageCAS CT -- direct comparison against each group's existing
"none" baseline result from the earlier mode sweeps.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

MANIFESTS = {
    "acdc": ROOT / "data/manifests_local/acdc_all.csv",
    "camus": ROOT / "data/manifests_local/camus_all.csv",
    "combined": ROOT / "data/manifests_local/segmentation_public_combined.csv",
    "imagecas": ROOT / "data/manifests_local/imagecas_1_200.csv",
}

CACHE_DIRS = {
    "acdc": ROOT / "outputs/_preprocess_cache_mmsweep",
    "camus": ROOT / "outputs/_preprocess_cache_mmsweep",
    "combined": ROOT / "outputs/_preprocess_cache_mmsweep",
    "imagecas": ROOT / "outputs/_preprocess_cache_imagecas",
}

OUT_ROOT = ROOT / "outputs/adaptive_gate_sweep"


def run_job(group: str) -> None:
    out_dir = OUT_ROOT / group
    if (out_dir / "summary.json").exists():
        print(f"[SKIP] {group} already has summary.json", flush=True)
        return
    cmd = [
        PY, "-m", "cardiac_image_system.experiments.train_adaptive_gate_unet",
        "--manifest", str(MANIFESTS[group]),
        "--output-dir", str(out_dir),
        "--epochs", "10",
        "--batch-size", "8",
        "--seed", "42",
        "--validation-seed", "42",
        "--preprocess-cache-dir", str(CACHE_DIRS[group]),
    ]
    print(f"\n=== RUN adaptive_gate/{group} ===", flush=True)
    print(" ".join(cmd), flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    dt = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f"=== adaptive_gate/{group}: {status} in {dt/60:.1f} min ===", flush=True)
    if result.returncode != 0:
        raise SystemExit(f"Job adaptive_gate/{group} failed with code {result.returncode}")


def main() -> None:
    order = ["camus", "imagecas", "acdc", "combined"]
    t_start = time.time()
    for group in order:
        run_job(group)
    print(f"\nALL DONE in {(time.time()-t_start)/60:.1f} min total", flush=True)


if __name__ == "__main__":
    main()
