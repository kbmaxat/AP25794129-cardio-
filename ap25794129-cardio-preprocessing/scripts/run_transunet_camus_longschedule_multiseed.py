"""Multi-seed confirmation for the TransUNet CAMUS long-schedule resolution (Table 9): the
single-seed result showed the 10-epoch reversal was a fixed-budget artifact, but rested on one
run per side. This adds two more seeds (11, 22) at the same long-schedule protocol, for
none/wavelet/nlm, so the resolution itself is not a single-run coincidence.

Resumable: skips any run whose output directory already has summary.json. The seed-42 runs from
outputs/transunet_camus_longschedule/ are reused as the third seed for this comparison.
"""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

PYTHON = r"C:\pyv\cardio_gpu\Scripts\python.exe"
REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "data/manifests_local/segmentation_public_combined.csv"
SESSION_ROOT = Path("outputs/transunet_camus_longschedule_multiseed")

MODES = ["none", "wavelet", "nlm"]
SEEDS = [11, 22]


def log(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def run_one(mode: str, seed: int) -> bool:
    output_dir = SESSION_ROOT / f"transunet_camus_{mode}_seed{seed}"
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        log(f"SKIP (already done): {output_dir}")
        return True

    args = [
        PYTHON, "-m", "cardiac_image_system.experiments.train_unet_baseline",
        "--manifest", MANIFEST,
        "--output-dir", str(output_dir),
        "--architecture", "transunet",
        "--mode", mode,
        "--dataset-filter", "CAMUS",
        "--epochs", "50",
        "--batch-size", "8",
        "--seed", str(seed),
        "--num-workers", "4",
        "--early-stopping-patience", "10",
        "--early-stopping-min-epochs", "15",
        "--early-stopping-min-delta", "0.0005",
    ]

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_dir.parent / f"{output_dir.name}.log"
    log(f"RUN {output_dir} (mode={mode}, seed={seed}, long-schedule)")

    with log_path.open("w", encoding="utf-8") as logf:
        result = subprocess.run(args, cwd=REPO_ROOT, stdout=logf, stderr=subprocess.STDOUT)

    ok = result.returncode == 0
    log(f"{'OK' if ok else 'FAILED'}: {output_dir} (log: {log_path})")
    return ok


def main() -> None:
    jobs = [(mode, seed) for seed in SEEDS for mode in MODES]
    log(f"Total jobs: {len(jobs)}")
    results = []
    for index, (mode, seed) in enumerate(jobs, start=1):
        log(f"=== Job {index}/{len(jobs)} ===")
        ok = run_one(mode, seed)
        results.append((f"{mode}_seed{seed}", ok))

    succeeded = sum(1 for _, ok in results if ok)
    log(f"Matrix complete: {succeeded}/{len(results)} succeeded.")
    failed = [name for name, ok in results if not ok]
    if failed:
        log("Failed runs:")
        for name in failed:
            log(f"  - {name}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
