# -*- coding: utf-8 -*-
"""Primary-budget (10-epoch, seed 42) six-mode preprocessing benchmark on the ImageCAS
cardiac CT dataset (patients 1-200 chunk, STACOM whole-heart labelmaps), mirroring the
primary grid already run on ACDC/CAMUS/combined. Confirms (or not) the dissertation's
central finding -- no preprocessing mode robustly beats raw input -- on a third, CT
modality, directly addressing the predzashita committee's question about modality
coverage beyond MRI/echo.
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
OUT_ROOT = ROOT / "outputs/imagecas_ct_primary_sweep"


def run_job(mode: str) -> None:
    out_dir = OUT_ROOT / mode
    if (out_dir / "summary.json").exists():
        print(f"[SKIP] {mode} already has summary.json", flush=True)
        return
    cmd = [
        PY, "-m", "cardiac_image_system.experiments.train_unet_baseline",
        "--manifest", str(MANIFEST),
        "--output-dir", str(out_dir),
        "--epochs", "10",
        "--batch-size", "8",
        "--seed", "42",
        "--validation-seed", "42",
        "--preprocess-cache-dir", str(CACHE_DIR),
        *MODES[mode],
    ]
    print(f"\n=== RUN imagecas/{mode} ===", flush=True)
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
