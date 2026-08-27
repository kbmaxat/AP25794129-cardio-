"""Confirmatory ACDC long-schedule sweep under the corrected, official-split pipeline.

Reruns the six-mode long-schedule ACDC confirmation (Table 5 / Table 6 in the manuscript)
under the Track A.1 split fix: the official ACDC 100-training/50-testing patients are now
used as-is (split_by_subset_column_carving_validation), with validation carved from the 100
training patients using a constant validation_seed independent of the training seed. This
replaces the manuscript's currently-reported 105/15/30 random-split ACDC results, whose
30-patient test set is not directly comparable to prior literature reporting on the official
50-patient partition (Limitations, "data-scope choices").

Uses the preprocessing cache (Track A.3) since nlm and hybrid are CPU-bound and this reruns
the same six (image, mode) combinations across up to 50 epochs each.

Writes into a new outputs/acdc_official_split_longschedule/ subtree; does not touch or
overwrite the existing outputs/phase2_full_longschedule/longschedule_acdc_all6 results that
the current manuscript reports. Resumable: a run is skipped if its output dir already has a
summary.json.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

PYTHON = r"C:\pyv\cardio_gpu\Scripts\python.exe"
REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "data/manifests_local/segmentation_public_combined.csv"
SESSION_ROOT = Path("outputs/acdc_official_split_longschedule")
CACHE_DIR = Path("outputs/_preprocess_cache")

ALL_MODES = ["none", "gaussian", "wavelet", "nlm", "clahe", "hybrid"]


def log(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def run_one(mode: str) -> bool:
    output_dir = SESSION_ROOT / f"unet_acdc_{mode}"
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        log(f"SKIP (already done): {output_dir}")
        return True

    args = [
        PYTHON, "-m", "cardiac_image_system.experiments.train_unet_baseline",
        "--manifest", MANIFEST,
        "--output-dir", str(output_dir),
        "--mode", mode,
        "--dataset-filter", "ACDC",
        "--epochs", "50",
        "--batch-size", "8",
        "--seed", "42",
        "--validation-seed", "42",
        "--num-workers", "4",
        "--early-stopping-patience", "10",
        "--early-stopping-min-epochs", "15",
        "--early-stopping-min-delta", "0.0005",
        "--preprocess-cache-dir", str(CACHE_DIR),
    ]

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_dir.parent / f"{output_dir.name}.log"
    log(f"RUN {output_dir} (mode={mode}, official ACDC split, long-schedule)")

    with log_path.open("w", encoding="utf-8") as logf:
        result = subprocess.run(args, cwd=REPO_ROOT, stdout=logf, stderr=subprocess.STDOUT)

    ok = result.returncode == 0
    log(f"{'OK' if ok else 'FAILED'}: {output_dir} (log: {log_path})")
    return ok


def main() -> None:
    log(f"Total jobs: {len(ALL_MODES)}")
    results = []
    for index, mode in enumerate(ALL_MODES, start=1):
        log(f"=== Job {index}/{len(ALL_MODES)} ===")
        ok = run_one(mode)
        results.append((mode, ok))

    succeeded = sum(1 for _, ok in results if ok)
    log(f"Matrix complete: {succeeded}/{len(results)} succeeded.")
    failed = [name for name, ok in results if not ok]
    if failed:
        log("Failed runs:")
        for name in failed:
            log(f"  - {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
