"""Third-architecture check: a genuinely non-U-Net-family backbone (TransUNet-style, with a
multi-head self-attention bottleneck) on the top-3 modes (none, wavelet, nlm) across all three
benchmark settings, at the same primary 10-epoch budget used for the Attention U-Net check, for
direct comparability across all three architectures now tested.

Resumable: skips any run whose output directory already has summary.json.
"""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

PYTHON = r"C:\pyv\cardio_gpu\Scripts\python.exe"
REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "data/manifests_local/segmentation_public_combined.csv"
SESSION_ROOT = Path("outputs/transunet_top3")

MODES = ["none", "wavelet", "nlm"]
DATASET_GROUPS: dict[str, list[str]] = {
    "acdc": ["ACDC"],
    "camus": ["CAMUS"],
    "combined": [],
}


def log(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def run_one(output_dir: Path, mode: str, dataset_filter: list[str], seed: int = 42, epochs: int = 10) -> bool:
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
        "--epochs", str(epochs),
        "--batch-size", "8",
        "--seed", str(seed),
        "--num-workers", "4",
    ]
    if dataset_filter:
        args += ["--dataset-filter", *dataset_filter]

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_dir.parent / f"{output_dir.name}.log"
    log(f"RUN {output_dir} (mode={mode}, dataset_filter={dataset_filter or 'ALL'})")

    with log_path.open("w", encoding="utf-8") as logf:
        result = subprocess.run(args, cwd=REPO_ROOT, stdout=logf, stderr=subprocess.STDOUT)

    ok = result.returncode == 0
    log(f"{'OK' if ok else 'FAILED'}: {output_dir} (log: {log_path})")
    return ok


def main() -> None:
    jobs = []
    for group_name, dataset_filter in DATASET_GROUPS.items():
        for mode in MODES:
            jobs.append(
                dict(
                    output_dir=SESSION_ROOT / group_name / f"transunet_{group_name}_{mode}",
                    mode=mode,
                    dataset_filter=dataset_filter,
                )
            )

    log(f"Total jobs: {len(jobs)}")
    results = []
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
        raise SystemExit(1)


if __name__ == "__main__":
    main()
