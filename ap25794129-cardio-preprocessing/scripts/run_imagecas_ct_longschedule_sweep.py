# -*- coding: utf-8 -*-
"""Long-schedule (up to 50 epochs, early stopping) confirmation of the primary-budget
ImageCAS CT six-mode grid, same protocol used throughout this benchmark. The raw-slice
disk cache is already warm from the primary sweep, so this pays only GPU training time.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

MANIFEST = ROOT / "data/manifests_local/imagecas_1_200.csv"

MODES = {
    "none": [],
    "gaussian": ["--mode", "gaussian"],
    "wavelet": ["--mode", "wavelet"],
    "nlm": ["--mode", "nlm"],
    "clahe": ["--mode", "clahe"],
    "hybrid": ["--mode", "hybrid"],
}

CACHE_DIR = ROOT / "outputs/_preprocess_cache_imagecas"
OUT_ROOT = ROOT / "outputs/imagecas_ct_longschedule_sweep"


def run_job(mode: str) -> None:
    out_dir = OUT_ROOT / mode
    if (out_dir / "summary.json").exists():
        print(f"[SKIP] {mode} already has summary.json", flush=True)
        return
    cmd = [
        PY, "-m", "cardiac_image_system.experiments.train_unet_baseline",
        "--manifest", str(MANIFEST),
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
    print(f"\n=== RUN imagecas/{mode} (long-schedule) ===", flush=True)
    print(" ".join(cmd), flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    dt = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f"=== imagecas/{mode}: {status} in {dt/60:.1f} min ===", flush=True)
    if result.returncode != 0:
        raise SystemExit(f"Job imagecas/{mode} failed with code {result.returncode}")


def main() -> None:
    order = ["none", "gaussian", "wavelet", "nlm", "clahe", "hybrid"]
    t_start = time.time()
    for mode in order:
        run_job(mode)
    print(f"\nALL DONE in {(time.time()-t_start)/60:.1f} min total", flush=True)


if __name__ == "__main__":
    main()
