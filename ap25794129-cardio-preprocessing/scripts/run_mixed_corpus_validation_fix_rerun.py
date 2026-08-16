"""Rerun every mixed-corpus (ACDC+CAMUS combined) training branch after fixing a real bug in
``split_by_subset_column_carving_validation``: for a pooled manifest where only one dataset
(CAMUS) supplies a native ``validation`` tag and the other (ACDC) does not, the old code treated
any non-empty validation set as sufficient and returned early, so checkpoint selection (best
validation loss / early stopping) for every mixed-corpus run in the manuscript was based on
CAMUS-only validation loss, with zero ACDC representation. Table 1's own numbers corroborated
this: mixed-corpus validation (50/200) exactly equalled CAMUS's standalone validation count.

The fix (cardiac_image_system/core/splits.py) now carves validation per-dataset for exactly the
datasets missing native coverage, verified by a new regression test
(test_carving_validation_carves_only_the_dataset_missing_native_validation) and the full existing
test suite (50/50 passing, no regressions).

This script reruns only the affected branches, on the combined manifest (dataset_filter=()),
matching each branch's original parameters exactly (epochs, batch size, seed(s), early-stopping
schedule, architecture) so the only thing that changes is the validation-set composition:
  1. Primary 10-epoch grid, all 6 modes (outputs/unet_mode_grid equivalent)
  2. Long-schedule (up to 50 epochs, early stopping), all 6 modes
  3. Five-seed multiseed at the 10-epoch budget, all 6 modes x 5 seeds
  4. Attention U-Net second-architecture check, top-3 modes, 10-epoch budget
  5. TransUNet third-architecture check, top-3 modes, 10-epoch budget

ACDC-only and CAMUS-only branches are unaffected by this bug (single dataset present, so the old
"any non-empty val" check was already correct there) and are not rerun.

Writes into a new outputs/mixed_corpus_validation_fix_rerun/ subtree; does not touch or overwrite
any existing outputs/ directory, so the original (bugged) mixed-corpus results remain available
for direct before/after comparison. Resumable: a run is skipped if its output dir already has a
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
SESSION_ROOT = Path("outputs/mixed_corpus_validation_fix_rerun")

ALL_MODES = ["none", "gaussian", "wavelet", "nlm", "clahe", "hybrid"]
TOP3_MODES = ["none", "wavelet", "nlm"]
MULTISEED_SEEDS = [11, 22, 33, 44, 55]
LONGSCHEDULE_KWARGS = dict(patience=10, min_epochs=15, min_delta=0.0005)


def log(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def run_one(
    output_dir: Path,
    mode: str,
    seed: int,
    epochs: int,
    architecture: str = "unet",
    early_stopping: dict | None = None,
    batch_size: int = 8,
    num_workers: int = 4,
) -> bool:
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        log(f"SKIP (already done): {output_dir}")
        return True

    args = [
        PYTHON, "-m", "cardiac_image_system.experiments.train_unet_baseline",
        "--manifest", MANIFEST,
        "--output-dir", str(output_dir),
        "--architecture", architecture,
        "--mode", mode,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--seed", str(seed),
        "--num-workers", str(num_workers),
    ]
    if early_stopping:
        args += [
            "--early-stopping-patience", str(early_stopping["patience"]),
            "--early-stopping-min-epochs", str(early_stopping["min_epochs"]),
            "--early-stopping-min-delta", str(early_stopping["min_delta"]),
        ]

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_dir.parent / f"{output_dir.name}.log"
    log(f"RUN {output_dir}  (architecture={architecture}, mode={mode}, seed={seed}, "
        f"epochs={epochs}, long_schedule={bool(early_stopping)})")

    with log_path.open("w", encoding="utf-8") as logf:
        result = subprocess.run(args, cwd=REPO_ROOT, stdout=logf, stderr=subprocess.STDOUT)

    ok = result.returncode == 0
    log(f"{'OK' if ok else 'FAILED'}: {output_dir} (log: {log_path})")
    return ok


def build_jobs() -> list[dict]:
    jobs: list[dict] = []

    for mode in ALL_MODES:
        jobs.append(dict(
            output_dir=SESSION_ROOT / "primary_grid" / f"unet_combined_{mode}",
            mode=mode, seed=42, epochs=10, early_stopping=None,
        ))

    for mode in ALL_MODES:
        jobs.append(dict(
            output_dir=SESSION_ROOT / "longschedule" / f"unet_combined_{mode}",
            mode=mode, seed=42, epochs=50, early_stopping=LONGSCHEDULE_KWARGS,
        ))

    for mode in ALL_MODES:
        for seed in MULTISEED_SEEDS:
            jobs.append(dict(
                output_dir=SESSION_ROOT / "multiseed" / f"unet_combined_{mode}_seed{seed}",
                mode=mode, seed=seed, epochs=10, early_stopping=None,
            ))

    for mode in TOP3_MODES:
        jobs.append(dict(
            output_dir=SESSION_ROOT / "attention_unet" / f"attn_unet_combined_{mode}",
            mode=mode, seed=42, epochs=10, architecture="attention_unet", early_stopping=None,
        ))

    for mode in TOP3_MODES:
        jobs.append(dict(
            output_dir=SESSION_ROOT / "transunet" / f"transunet_combined_{mode}",
            mode=mode, seed=42, epochs=10, architecture="transunet", early_stopping=None,
        ))

    return jobs


def main() -> None:
    jobs = build_jobs()
    log(f"Total jobs in mixed-corpus validation-fix rerun matrix: {len(jobs)}")

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
