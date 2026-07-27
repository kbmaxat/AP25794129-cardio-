from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "unet_mode_grid"

SESSION_MAP = {
    "ACDC": OUTPUT_ROOT / "20260701_105111",
    "CAMUS": OUTPUT_ROOT / "20260701_105750_camus",
    "COMBINED": OUTPUT_ROOT / "20260701_105750_combined",
}


def load_session_rows(dataset_label: str, session_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run_dir in sorted(session_root.iterdir()):
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "dataset_group": dataset_label,
                "session_root": str(session_root),
                "run_name": run_dir.name,
                "mode": data["config"]["preprocess_mode"],
                "device": data["device"],
                "parameter_count": data["parameter_count"],
                "train_rows": data["train_rows"],
                "val_rows": data["val_rows"],
                "test_rows": data["test_rows"],
                "train_patients": data["train_patients"],
                "val_patients": data["val_patients"],
                "test_patients": data["test_patients"],
                "best_val_loss": data["best_val_loss"],
                "test_loss": data["test_loss"],
                "dice_mean": data["test_metrics_slice_mean"]["dice"],
                "iou_mean": data["test_metrics_slice_mean"]["iou"],
                "hd95_mean": data["test_metrics_slice_mean"]["hd95"],
                "inference_ms_per_image": data["test_inference_ms_per_image"],
            }
        )
    return rows


def main() -> None:
    all_rows: list[dict[str, object]] = []
    for dataset_label, session_root in SESSION_MAP.items():
        all_rows.extend(load_session_rows(dataset_label, session_root))

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise SystemExit("No summary rows found.")

    df = df.sort_values(["dataset_group", "dice_mean"], ascending=[True, False]).reset_index(drop=True)
    best_df = df.groupby("dataset_group", as_index=False).first()

    out_csv = OUTPUT_ROOT / "public_segmentation_suite_summary.csv"
    out_best_csv = OUTPUT_ROOT / "public_segmentation_suite_best_by_dataset.csv"
    out_md = OUTPUT_ROOT / "public_segmentation_suite_summary.md"

    df.to_csv(out_csv, index=False)
    best_df.to_csv(out_best_csv, index=False)

    lines: list[str] = []
    lines.append("# Public Segmentation Suite Summary")
    lines.append("")
    for dataset_label in ["ACDC", "CAMUS", "COMBINED"]:
        subset = df[df["dataset_group"] == dataset_label].copy()
        if subset.empty:
            continue
        lines.append(f"## {dataset_label}")
        lines.append("")
        lines.append("| Mode | Dice | IoU | HD95 | Inference (ms/image) |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, row in subset.iterrows():
            lines.append(
                f"| {row['mode']} | {row['dice_mean']:.4f} | {row['iou_mean']:.4f} | "
                f"{row['hd95_mean']:.4f} | {row['inference_ms_per_image']:.4f} |"
            )
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_best_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
