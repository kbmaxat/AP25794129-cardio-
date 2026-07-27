from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cardiac_image_system.core.datasets import (
    build_acdc_manifest,
    build_camus_manifest,
    summarize_manifest_by_group,
)
from cardiac_image_system.core.manifest import summarize_manifest
from cardiac_image_system.core.splits import export_split_manifests, split_by_subset_column


def _default_workspace_root(repo_root: Path) -> Path:
    return repo_root.parent.parent


def build_bundle(
    acdc_root: Path,
    camus_root: Path,
    output_dir: Path,
    include_empty_acdc_masks: bool = False,
    include_camus_half_sequences: bool = False,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)

    acdc_df = build_acdc_manifest(acdc_root, include_empty_masks=include_empty_acdc_masks)
    camus_df = build_camus_manifest(camus_root, include_half_sequences=include_camus_half_sequences)
    combined_df = pd.concat([acdc_df, camus_df], ignore_index=True)

    manifests = {
        "acdc_all": acdc_df,
        "camus_all": camus_df,
        "segmentation_public_combined": combined_df,
    }
    for name, df in manifests.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)

    export_split_manifests(split_by_subset_column(acdc_df), output_dir, prefix="acdc")
    export_split_manifests(split_by_subset_column(camus_df), output_dir, prefix="camus")

    group_summary = summarize_manifest_by_group(combined_df)
    group_summary.to_csv(output_dir / "dataset_group_summary.csv", index=False)

    summary = {
        "acdc_all": summarize_manifest(acdc_df),
        "camus_all": summarize_manifest(camus_df),
        "combined": summarize_manifest(combined_df),
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return manifests


def main() -> None:
    repo_root = REPO_ROOT
    workspace_root = _default_workspace_root(repo_root)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--acdc-root",
        type=Path,
        default=workspace_root / "data" / "segmentation" / "ACDC_full" / "database",
    )
    parser.add_argument(
        "--camus-root",
        type=Path,
        default=workspace_root / "data" / "segmentation" / "CAMUS_full",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "data" / "manifests_local",
    )
    parser.add_argument("--include-empty-acdc-masks", action="store_true")
    parser.add_argument("--include-camus-half-sequences", action="store_true")
    args = parser.parse_args()

    manifests = build_bundle(
        acdc_root=args.acdc_root,
        camus_root=args.camus_root,
        output_dir=args.output_dir,
        include_empty_acdc_masks=args.include_empty_acdc_masks,
        include_camus_half_sequences=args.include_camus_half_sequences,
    )
    for name, df in manifests.items():
        print(f"{name}: rows={len(df)} patients={df['patient_id'].nunique()}")


if __name__ == "__main__":
    main()
