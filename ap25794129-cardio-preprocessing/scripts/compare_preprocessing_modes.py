"""Reproduce a table3-style paired mode-vs-baseline comparison from real run outputs.

Each ``--run`` argument points at one preprocessing-mode's experiment output directory
(as produced by ``train_unet_baseline`` / ``train_unet_multiclass``), which must contain
a ``test_patient_level.csv`` file with a ``patient_id`` column and the requested metric
column. This script pairs patients across modes, then reports the bootstrap CI, paired
Wilcoxon test, Holm-adjusted p-value, and a TOST equivalence check for each non-baseline
mode relative to the baseline mode.

Example:

    python scripts/compare_preprocessing_modes.py \\
        --run none=outputs/unet_baseline_combined_none \\
        --run wavelet=outputs/unet_baseline_combined_wavelet \\
        --run nlm=outputs/unet_baseline_combined_nlm \\
        --baseline-mode none \\
        --metric dice \\
        --output-csv outputs/mode_comparison_dice.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cardiac_image_system.core.stats import compare_modes_to_baseline


def _parse_run_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"--run must be MODE=PATH, got: {raw}")
    mode, path = raw.split("=", 1)
    return mode, Path(path)


def load_patient_metric(run_dir: Path, metric: str) -> pd.Series:
    csv_path = run_dir / "test_patient_level.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected patient-level results at: {csv_path}")
    df = pd.read_csv(csv_path)
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in {csv_path}. Available columns: {list(df.columns)}")
    return df.set_index("patient_id")[metric]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, type=_parse_run_arg, help="MODE=PATH, repeatable")
    parser.add_argument("--baseline-mode", required=True)
    parser.add_argument("--metric", default="dice")
    parser.add_argument("--equivalence-margin", type=float, default=0.01)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()

    patient_metric_by_mode = {mode: load_patient_metric(path, args.metric) for mode, path in args.run}

    result = compare_modes_to_baseline(
        patient_metric_by_mode,
        baseline_mode=args.baseline_mode,
        equivalence_margin=args.equivalence_margin,
        n_bootstrap=args.n_bootstrap,
        ci_level=args.ci_level,
        alpha=args.alpha,
        seed=args.seed,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    print(result.to_string(index=False))
    print(f"\nWrote {args.output_csv}")


if __name__ == "__main__":
    main()
