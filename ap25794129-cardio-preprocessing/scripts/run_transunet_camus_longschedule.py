"""Resolve the fixed-budget-vs-architecture confound in the TransUNet CAMUS reversal: rerun
none/wavelet/nlm under TransUNet on CAMUS with the same long-schedule, early-stopping protocol
(up to 50 epochs) used elsewhere in this benchmark. If raw-input Dice recovers toward the
U-Net-family range and the wavelet/nlm advantage shrinks or disappears, the reversal was a
fixed-budget artifact of the primary 10-epoch check. If it persists, the reversal is more likely a
stable property of this architecture on this modality.

Resumable: skips any run whose output directory already has summary.json.
"""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

PYTHON = r"C:\pyv\cardio_gpu\Scripts\python.exe"
REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "data/manifests_local/segmentation_public_combined.csv"
SESSION_ROOT = Path("outputs/transunet_camus_longschedule")

MODES = ["none", "wavelet", "nlm"]


def log(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def run_one(mode: str) -> bool:
    output_dir = SESSION_ROOT / f"transunet_camus_{mode}"
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
        "--seed", "42",
        "--num-workers", "4",
        "--early-stopping-patience", "10",
        "--early-stopping-min-epochs", "15",
        "--early-stopping-min-delta", "0.0005",
    ]

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_dir.parent / f"{output_dir.name}.log"
    log(f"RUN {output_dir} (mode={mode}, long-schedule)")

    with log_path.open("w", encoding="utf-8") as logf:
        result = subprocess.run(args, cwd=REPO_ROOT, stdout=logf, stderr=subprocess.STDOUT)

    ok = result.returncode == 0
    log(f"{'OK' if ok else 'FAILED'}: {output_dir} (log: {log_path})")
    return ok


def main() -> None:
    log(f"Total jobs: {len(MODES)}")
    results = []
    for index, mode in enumerate(MODES, start=1):
        log(f"=== Job {index}/{len(MODES)} ===")
        ok = run_one(mode)
        results.append((mode, ok))

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
