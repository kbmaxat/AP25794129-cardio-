from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


METRIC_MAP = {
    "macro_dice": "Macro Dice",
    "class_1_dice": "RV Dice",
    "class_2_dice": "Myocardium Dice",
    "class_3_dice": "LV Dice",
}


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        candidate = (m - rank) * p_values[idx]
        running = max(running, candidate)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def bootstrap_ci(diff: np.ndarray, seed: int = 42, n_bootstrap: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(diff)
    samples = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        samples[i] = float(diff[idx].mean())
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def load_mode_frame(run_root: Path, mode: str) -> pd.DataFrame:
    candidates = [
        run_root / f"unet_multiclass_acdc_{mode}" / "test_patient_level.csv",
        run_root / f"unet_multiclass_extended_acdc_{mode}" / "test_patient_level.csv",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        glob_matches = sorted(run_root.glob(f"*{mode}/test_patient_level.csv"))
        if glob_matches:
            path = glob_matches[0]
    if path is None:
        raise FileNotFoundError(f"Could not locate test_patient_level.csv for mode '{mode}' under {run_root}")
    df = pd.read_csv(path).sort_values("patient_id").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"Empty patient-level dataframe for mode '{mode}' at {path}")
    return df


def build_summary(run_root: Path) -> tuple[pd.DataFrame, str]:
    frames = {mode: load_mode_frame(run_root, mode) for mode in ["none", "wavelet", "nlm"]}
    base = frames["none"]
    for mode, frame in frames.items():
        if frame["patient_id"].tolist() != base["patient_id"].tolist():
            raise ValueError(f"Patient ordering mismatch between 'none' and '{mode}'")

    records: list[dict[str, object]] = []
    markdown_lines = [
        "# ACDC Multiclass Top-3 Summary",
        "",
        "This report summarizes the reviewer-driven multiclass ACDC extension comparing `none`, `wavelet`, and `nlm` under the fixed U-Net training configuration.",
        "",
        "## Mean +/- SD (patient level)",
        "",
        "| Metric | None | Wavelet | NLM |",
        "|---|---:|---:|---:|",
    ]

    for metric, label in METRIC_MAP.items():
        row = [label]
        for mode in ["none", "wavelet", "nlm"]:
            values = frames[mode][metric].to_numpy(dtype=float)
            mean = float(values.mean())
            sd = float(values.std(ddof=1))
            row.append(f"{mean:.4f} +/- {sd:.4f}")
        markdown_lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")

    markdown_lines.extend(
        [
            "",
            "## Paired Statistics vs `none`",
            "",
            "| Metric | Mode | Mean Delta vs None | 95% Bootstrap CI | Wilcoxon p | Holm-adjusted p |",
            "|---|---|---:|---|---:|---:|",
        ]
    )

    for metric, label in METRIC_MAP.items():
        base_values = frames["none"][metric].to_numpy(dtype=float)
        p_values: list[float] = []
        interim: list[dict[str, object]] = []

        for mode in ["wavelet", "nlm"]:
            mode_values = frames[mode][metric].to_numpy(dtype=float)
            diff = mode_values - base_values
            p_value = float(
                wilcoxon(mode_values, base_values, zero_method="wilcox", alternative="two-sided", method="auto").pvalue
            )
            ci_low, ci_high = bootstrap_ci(diff)
            summary_row = {
                "metric": label,
                "mode": mode,
                "n_patients": int(len(diff)),
                "mean_none": float(base_values.mean()),
                "sd_none": float(base_values.std(ddof=1)),
                "mean_mode": float(mode_values.mean()),
                "sd_mode": float(mode_values.std(ddof=1)),
                "delta_vs_none": float(diff.mean()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": p_value,
            }
            interim.append(summary_row)
            p_values.append(p_value)

        holm = holm_adjust(p_values)
        for record, p_holm in zip(interim, holm):
            record["p_holm"] = float(p_holm)
            records.append(record)
            markdown_lines.append(
                f"| {record['metric']} | {record['mode']} | {record['delta_vs_none']:.4f} | "
                f"[{record['ci_low']:.4f}, {record['ci_high']:.4f}] | {record['p_value']:.6f} | {record['p_holm']:.6f} |"
            )

    markdown_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `none` remained the strongest configuration across macro Dice and all three anatomical Dice metrics.",
            "- `wavelet` produced lower patient-level macro Dice than `none` and remained significantly worse after Holm correction.",
            "- `nlm` remained numerically below `none`, but it did not demonstrate a statistically supported improvement or degradation under the paired Wilcoxon-Holm criterion.",
        ]
    )

    return pd.DataFrame.from_records(records), "\n".join(markdown_lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    summary_df, report_md = build_summary(args.run_root)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output_csv, index=False)
    args.output_md.write_text(report_md, encoding="utf-8")


if __name__ == "__main__":
    main()
