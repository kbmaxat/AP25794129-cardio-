"""Minimal preprocessing-hyperparameter sensitivity analysis, per reviewer request: the
"no preprocessing benefit" finding is strictly about the specific wavelet/NLM/CLAHE parameter
values tested throughout the rest of this benchmark. This checks two additional values per
parameter (below and above the default), on ACDC only, at the primary 10-epoch budget, for the
one mode each parameter belongs to.

Resumable: skips any run whose output directory already has summary.json.
"""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

PYTHON = r"C:\pyv\cardio_gpu\Scripts\python.exe"
REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "data/manifests_local/segmentation_public_combined.csv"
SESSION_ROOT = Path("outputs/sensitivity_analysis_acdc")

# (mode, param_flag, param_value, run_tag)
JOBS = [
    ("wavelet", "--wavelet-level", "1", "wavelet_level1"),
    ("wavelet", "--wavelet-level", "3", "wavelet_level3"),
    ("nlm", "--nlm-h-multiplier", "0.4", "nlm_hmult0.4"),
    ("nlm", "--nlm-h-multiplier", "1.6", "nlm_hmult1.6"),
    ("clahe", "--clahe-clip-limit", "0.01", "clahe_clip0.01"),
    ("clahe", "--clahe-clip-limit", "0.06", "clahe_clip0.06"),
]


def log(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def run_one(mode: str, param_flag: str, param_value: str, run_tag: str) -> bool:
    output_dir = SESSION_ROOT / f"unet_acdc_{run_tag}"
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
        "--epochs", "10",
        "--batch-size", "8",
        "--seed", "42",
        "--num-workers", "4",
        param_flag, param_value,
    ]

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_dir.parent / f"{output_dir.name}.log"
    log(f"RUN {output_dir} (mode={mode}, {param_flag}={param_value})")

    with log_path.open("w", encoding="utf-8") as logf:
        result = subprocess.run(args, cwd=REPO_ROOT, stdout=logf, stderr=subprocess.STDOUT)

    ok = result.returncode == 0
    log(f"{'OK' if ok else 'FAILED'}: {output_dir} (log: {log_path})")
    return ok


def main() -> None:
    log(f"Total jobs: {len(JOBS)}")
    results = []
    for index, (mode, flag, value, tag) in enumerate(JOBS, start=1):
        log(f"=== Job {index}/{len(JOBS)} ===")
        ok = run_one(mode, flag, value, tag)
        results.append((tag, ok))

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
