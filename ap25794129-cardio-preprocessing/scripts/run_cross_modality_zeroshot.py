"""Zero-shot cross-modality evaluation: apply a checkpoint trained on one dataset group
(ACDC or CAMUS) directly to the other's held-out test set, with no fine-tuning.

This is deliberately distinct from the mixed-corpus (ACDC+CAMUS) benchmark elsewhere in this
repository, which trains on a pooled corpus and therefore does not test transfer across domains.
Here the model never sees the target modality during training.

Takes ``--target-manifest`` as an already-resolved test-split CSV (e.g. the
``resolved_splits/unet_baseline_test.csv`` written by train_unet_baseline.py for the target
dataset's own run) rather than re-deriving a split, so the exact same held-out patient cohort
used throughout the rest of the benchmark is reused here too -- no new split logic, no risk of
divergent leakage-prevention behavior.

Example:

    python scripts/run_cross_modality_zeroshot.py \\
      --checkpoint outputs/phase2_full_longschedule/longschedule_acdc_all6/unet_acdc_none/checkpoint_best.pt \\
      --target-manifest outputs/phase2_full_longschedule/longschedule_camus_remaining3/unet_camus_none/resolved_splits/unet_baseline_test.csv \\
      --mode none \\
      --output-dir outputs/cross_modality_zeroshot/acdc_to_camus_none
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cardiac_image_system.core.manifest import load_manifest
from cardiac_image_system.core.metrics import dice, hd95, iou
from cardiac_image_system.core.torch_data import ManifestSegmentationDataset
from cardiac_image_system.core.validation import aggregate_patient_level
from cardiac_image_system.experiments.train_unet_baseline import ARCHITECTURES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--architecture", choices=sorted(ARCHITECTURES), default="unet")
    parser.add_argument("--target-manifest", required=True, type=Path)
    parser.add_argument("--mode", default="none")
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    target = load_manifest(args.target_manifest)
    if target.empty:
        raise SystemExit(f"Target manifest is empty: {args.target_manifest}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ARCHITECTURES[args.architecture]().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    dataset = ManifestSegmentationDataset(
        manifest=target,
        image_size=(args.image_height, args.image_width),
        preprocess_mode=args.mode,
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    rows: list[dict[str, object]] = []
    inference_times_ms: list[float] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            start = time.perf_counter()
            logits = model(images)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            inference_times_ms.append(elapsed_ms / max(int(images.size(0)), 1))

            probs = torch.sigmoid(logits).cpu().numpy()
            true_masks = masks.cpu().numpy()
            for index in range(probs.shape[0]):
                pred_mask = probs[index, 0] >= args.threshold
                true_mask = true_masks[index, 0] >= 0.5
                rows.append(
                    {
                        "patient_id": batch["patient_id"][index],
                        "mode": args.mode,
                        "phase": batch["phase"][index],
                        "dataset": batch["dataset"][index],
                        "subset": batch["subset"][index],
                        "dice": dice(true_mask, pred_mask),
                        "iou": iou(true_mask, pred_mask),
                        "hd95": hd95(true_mask, pred_mask),
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    slice_df = pd.DataFrame(rows)
    slice_df.to_csv(args.output_dir / "test_slice_level.csv", index=False)
    patient_df = aggregate_patient_level(slice_df)
    patient_df.to_csv(args.output_dir / "test_patient_level.csv", index=False)

    summary = {
        "checkpoint": str(args.checkpoint),
        "architecture": args.architecture,
        "target_manifest": str(args.target_manifest),
        "mode": args.mode,
        "test_patients": int(patient_df["patient_id"].nunique()) if not patient_df.empty else 0,
        "test_rows": int(len(slice_df)),
        "test_inference_ms_per_image": float(np.mean(inference_times_ms)) if inference_times_ms else float("nan"),
        "test_metrics_slice_mean": {
            "dice": float(slice_df["dice"].mean()) if not slice_df.empty else float("nan"),
            "iou": float(slice_df["iou"].mean()) if not slice_df.empty else float("nan"),
            "hd95": float(slice_df["hd95"].replace([np.inf, -np.inf], np.nan).mean()) if not slice_df.empty else float("nan"),
        },
        "test_metrics_patient_mean": {
            "dice": float(patient_df["dice"].mean()) if not patient_df.empty else float("nan"),
            "iou": float(patient_df["iou"].mean()) if not patient_df.empty else float("nan"),
            "hd95": float(patient_df["hd95"].replace([np.inf, -np.inf], np.nan).mean()) if not patient_df.empty else float("nan"),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
