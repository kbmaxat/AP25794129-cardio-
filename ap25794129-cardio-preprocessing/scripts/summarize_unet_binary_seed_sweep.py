from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_rows(session_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for summary_path in sorted(session_root.rglob("summary.json")):
        run_dir = summary_path.parent
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        config = data.get("config", {})
        dataset_filter = config.get("dataset_filter") or []
        if dataset_filter:
            dataset_group = "+".join(str(x) for x in dataset_filter)
        else:
            dataset_group = "ACDC+CAMUS"

        rows.append(
            {
                "dataset_group": dataset_group,
                "mode": config.get("preprocess_mode", "unknown"),
                "seed": config.get("seed"),
                "run_dir": str(run_dir),
                "completed_epochs": data.get("completed_epochs"),
                "best_epoch": data.get("best_epoch"),
                "stopped_early": data.get("stopped_early"),
                "best_val_loss": data.get("best_val_loss"),
                "test_loss": data.get("test_loss"),
                "patient_dice_mean": data.get("test_metrics_patient_mean", {}).get("dice"),
                "patient_iou_mean": data.get("test_metrics_patient_mean", {}).get("iou"),
                "patient_hd95_mean": data.get("test_metrics_patient_mean", {}).get("hd95"),
                "slice_dice_mean": data.get("test_metrics_slice_mean", {}).get("dice"),
                "slice_iou_mean": data.get("test_metrics_slice_mean", {}).get("iou"),
                "slice_hd95_mean": data.get("test_metrics_slice_mean", {}).get("hd95"),
                "inference_ms_per_image": data.get("test_inference_ms_per_image"),
            }
        )
    return rows


def build_markdown(agg_df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# Binary Seed-Sweep Summary")
    lines.append("")
    for dataset_group in sorted(agg_df["dataset_group"].unique()):
        subset = agg_df[agg_df["dataset_group"] == dataset_group].copy()
        lines.append(f"## {dataset_group}")
        lines.append("")
        lines.append("| Mode | Runs | Patient Dice (mean ± SD) | Patient IoU (mean ± SD) | Patient HD95 (mean ± SD) | Mean Best Epoch | Early-Stopped Runs |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for _, row in subset.iterrows():
            lines.append(
                f"| {row['mode']} | {int(row['n_runs'])} | "
                f"{row['patient_dice_mean_mean']:.4f} ± {row['patient_dice_mean_std']:.4f} | "
                f"{row['patient_iou_mean_mean']:.4f} ± {row['patient_iou_mean_std']:.4f} | "
                f"{row['patient_hd95_mean_mean']:.4f} ± {row['patient_hd95_mean_std']:.4f} | "
                f"{row['best_epoch_mean']:.1f} | {int(row['stopped_early_sum'])} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-root", required=True, type=Path)
    parser.add_argument("--output-prefix", default="binary_seed_sweep", type=str)
    args = parser.parse_args()

    rows = load_rows(args.session_root)
    if not rows:
        raise SystemExit(f"No summary.json files found under {args.session_root}")

    df = pd.DataFrame(rows).sort_values(["dataset_group", "mode", "seed"]).reset_index(drop=True)

    agg_df = (
        df.groupby(["dataset_group", "mode"], as_index=False)
        .agg(
            n_runs=("seed", "count"),
            patient_dice_mean_mean=("patient_dice_mean", "mean"),
            patient_dice_mean_std=("patient_dice_mean", "std"),
            patient_iou_mean_mean=("patient_iou_mean", "mean"),
            patient_iou_mean_std=("patient_iou_mean", "std"),
            patient_hd95_mean_mean=("patient_hd95_mean", "mean"),
            patient_hd95_mean_std=("patient_hd95_mean", "std"),
            best_epoch_mean=("best_epoch", "mean"),
            completed_epochs_mean=("completed_epochs", "mean"),
            stopped_early_sum=("stopped_early", "sum"),
            inference_ms_mean=("inference_ms_per_image", "mean"),
            inference_ms_std=("inference_ms_per_image", "std"),
        )
        .sort_values(["dataset_group", "patient_dice_mean_mean"], ascending=[True, False])
        .reset_index(drop=True)
    )

    numeric_fill_cols = [
        "patient_dice_mean_std",
        "patient_iou_mean_std",
        "patient_hd95_mean_std",
        "inference_ms_std",
    ]
    agg_df[numeric_fill_cols] = agg_df[numeric_fill_cols].fillna(0.0)

    detail_csv = args.session_root / f"{args.output_prefix}_detail.csv"
    agg_csv = args.session_root / f"{args.output_prefix}_aggregate.csv"
    summary_md = args.session_root / f"{args.output_prefix}_summary.md"

    df.to_csv(detail_csv, index=False)
    agg_df.to_csv(agg_csv, index=False)
    summary_md.write_text(build_markdown(agg_df), encoding="utf-8")

    print(f"Wrote {detail_csv}")
    print(f"Wrote {agg_csv}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
