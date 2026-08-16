"""Phase 2: close the "full six-mode long-schedule/multiseed" gap from the
manuscript's own Limitations section (items 6 and 7).

Already done before this script (see outputs/unet_mode_grid, unet_binary_longschedule_*_top3,
unet_binary_multiseed_*_top3): primary 10-epoch 6-mode grid for ACDC/CAMUS/combined, and
long-schedule + multiseed for the top-3 modes (none, wavelet, nlm) on CAMUS and combined only.

Missing, and covered here:
  - ACDC: long-schedule (all 6 modes) and multiseed (all 6 modes x 5 seeds) in BINARY form
    (only the multiclass ACDC top-3 branch had a long-schedule rerun before this).
  - CAMUS / combined: long-schedule and multiseed for the 3 remaining modes
    (gaussian, clahe, hybrid) that never got either treatment.

Writes into a new outputs/phase2_full_longschedule/ subtree; does not touch or overwrite
any existing outputs/ directory. Resumable: a run is skipped if its output dir already has
a summary.json (so this script can be safely re-launched after an interruption).
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

PYTHON = r"C:\pyv\cardio_gpu\Scripts\python.exe"
REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "data/manifests_local/segmentation_public_combined.csv"
SESSION_ROOT = Path("outputs/phase2_full_longschedule")

ALL_MODES = ["none", "gaussian", "wavelet", "nlm", "clahe", "hybrid"]
REMAINING_MODES = ["gaussian", "clahe", "hybrid"]  # none/wavelet/nlm already covered elsewhere
MULTISEED_SEEDS = [11, 22, 33, 44, 55]
LONGSCHEDULE_KWARGS = dict(patience=10, min_epochs=15, min_delta=0.0005)


def log(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def run_one(output_dir: Path, mode: str, dataset_filter: list[str], seed: int, epochs: int,
            early_stopping: dict | None, batch_size: int = 8, num_workers: int = 4) -> bool:
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        log(f"SKIP (already done): {output_dir}")
        return True

    args = [
        PYTHON, "-m", "cardiac_image_system.experiments.train_unet_baseline",
        "--manifest", MANIFEST,
        "--output-dir", str(output_dir),
        "--mode", mode,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--seed", str(seed),
        "--num-workers", str(num_workers),
    ]
    if dataset_filter:
        args += ["--dataset-filter", *dataset_filter]
    if early_stopping:
        args += [
            "--early-stopping-patience", str(early_stopping["patience"]),
            "--early-stopping-min-epochs", str(early_stopping["min_epochs"]),
            "--early-stopping-min-delta", str(early_stopping["min_delta"]),
        ]

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_dir.parent / f"{output_dir.name}.log"
    log(f"RUN {output_dir}  (mode={mode}, dataset_filter={dataset_filter or 'ALL'}, "
        f"seed={seed}, epochs={epochs}, long_schedule={bool(early_stopping)})")

    with log_path.open("w", encoding="utf-8") as logf:
        result = subprocess.run(args, cwd=REPO_ROOT, stdout=logf, stderr=subprocess.STDOUT)

    ok = result.returncode == 0
    log(f"{'OK' if ok else 'FAILED'}: {output_dir} (log: {log_path})")
    return ok


def build_jobs() -> list[dict]:
    jobs: list[dict] = []

    for mode in ALL_MODES:
        jobs.append(dict(
            output_dir=SESSION_ROOT / "longschedule_acdc_all6" / f"unet_acdc_{mode}",
            mode=mode, dataset_filter=["ACDC"], seed=42, epochs=50,
            early_stopping=LONGSCHEDULE_KWARGS,
        ))

    for mode in ALL_MODES:
        for seed in MULTISEED_SEEDS:
            jobs.append(dict(
                output_dir=SESSION_ROOT / "multiseed_acdc_all6" / f"unet_acdc_{mode}_seed{seed}",
                mode=mode, dataset_filter=["ACDC"], seed=seed, epochs=10,
                early_stopping=None,
            ))

    for mode in REMAINING_MODES:
        jobs.append(dict(
            output_dir=SESSION_ROOT / "longschedule_camus_remaining3" / f"unet_camus_{mode}",
            mode=mode, dataset_filter=["CAMUS"], seed=42, epochs=50,
            early_stopping=LONGSCHEDULE_KWARGS,
        ))

    for mode in REMAINING_MODES:
        for seed in MULTISEED_SEEDS:
            jobs.append(dict(
                output_dir=SESSION_ROOT / "multiseed_camus_remaining3" / f"unet_camus_{mode}_seed{seed}",
                mode=mode, dataset_filter=["CAMUS"], seed=seed, epochs=10,
                early_stopping=None,
            ))

    for mode in REMAINING_MODES:
        jobs.append(dict(
            output_dir=SESSION_ROOT / "longschedule_combined_remaining3" / f"unet_combined_{mode}",
            mode=mode, dataset_filter=[], seed=42, epochs=50,
            early_stopping=LONGSCHEDULE_KWARGS,
        ))

    for mode in REMAINING_MODES:
        for seed in MULTISEED_SEEDS:
            jobs.append(dict(
                output_dir=SESSION_ROOT / "multiseed_combined_remaining3" / f"unet_combined_{mode}_seed{seed}",
                mode=mode, dataset_filter=[], seed=seed, epochs=10,
                early_stopping=None,
            ))

    return jobs


def main() -> None:
    jobs = build_jobs()
    log(f"Total jobs in matrix: {len(jobs)}")

    results: list[tuple[str, bool]] = []
    for index, job in enumerate(jobs, start=1):
        log(f"=== Job {index}/{len(jobs)}: {job['output_dir']} ===")
        ok = run_one(**job)
        results.append((str(job["output_dir"]), ok))

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
