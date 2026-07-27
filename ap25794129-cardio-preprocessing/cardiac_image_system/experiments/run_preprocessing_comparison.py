from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from cardiac_image_system.core.io import load_grayscale_image, save_grayscale_image
from cardiac_image_system.core.manifest import load_manifest, summarize_manifest
from cardiac_image_system.core.metrics import dice, hd95, iou, psnr, relative_area_error, ssim
from cardiac_image_system.core.preprocessing import preprocess_image
from cardiac_image_system.core.segmentation import otsu_proxy_segmentation
from cardiac_image_system.core.validation import aggregate_patient_level, save_runtime_log


def run_experiment(manifest_path: Path, output_dir: Path, modes: list[str]) -> pd.DataFrame:
    df = load_manifest(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = output_dir / "processed"
    masks_dir = output_dir / "masks"
    processed_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)

    rows = []
    for _, row in df.iterrows():
        patient_id = str(row["patient_id"])
        phase = str(row["phase"])
        slice_index = None
        if "slice_index" in row and not pd.isna(row["slice_index"]):
            slice_index = int(row["slice_index"])

        image = load_grayscale_image(row["image_path"], slice_index=slice_index)
        mask_true = load_grayscale_image(row["mask_path"], slice_index=slice_index) > 0.5

        for mode in modes:
            processed = preprocess_image(image, mode=mode)
            mask_pred = otsu_proxy_segmentation(processed)

            stem = f"{patient_id}_{phase}_{mode}"
            processed_path = processed_dir / f"{stem}.png"
            mask_path = masks_dir / f"{stem}_mask.png"
            save_grayscale_image(processed_path, processed)
            save_grayscale_image(mask_path, mask_pred.astype(float))

            result_row = {
                "patient_id": patient_id,
                "phase": phase,
                "mode": mode,
                "image_path": str(row["image_path"]),
                "processed_path": str(processed_path),
                "mask_pred_path": str(mask_path),
                "psnr": psnr(image, processed),
                "ssim": ssim(image, processed),
                "dice": dice(mask_true, mask_pred),
                "iou": iou(mask_true, mask_pred),
                "hd95": hd95(mask_true, mask_pred),
                "relative_area_error": relative_area_error(mask_true, mask_pred),
            }
            for key in ["dataset", "subset", "modality", "view", "slice_index", "frame_id", "group", "source_patient_id"]:
                if key in row.index:
                    result_row[key] = row[key]
            rows.append(result_row)

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "metrics_slice_level.csv", index=False)
    patient_results = aggregate_patient_level(results)
    patient_results.to_csv(output_dir / "metrics_patient_level.csv", index=False)

    save_runtime_log(
        output_dir,
        {
            "timestamp": datetime.now().isoformat(),
            "manifest": str(manifest_path),
            "modes": ",".join(modes),
            **summarize_manifest(df),
        },
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--modes", nargs="+", default=["none", "gaussian", "wavelet", "nlm", "clahe", "hybrid"])
    args = parser.parse_args()
    run_experiment(args.manifest, args.output_dir, args.modes)


if __name__ == "__main__":
    main()
