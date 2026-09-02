"""Build a manifest CSV for the ImageCAS cardiac CT dataset (1-200 chunk) using the
STACOM 2025 whole-heart labelmaps (Hansen et al., 2025) as segmentation targets."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cardiac_image_system.core.datasets import build_imagecas_manifest
from cardiac_image_system.core.manifest import summarize_manifest

IMAGE_ROOT = r"C:\Users\maksa\Desktop\диссертация\data\data\segmentation\ImageCAS_CT\raw_images\final_extracted\1-200"
SEG_ROOT = r"C:\Users\maksa\Desktop\диссертация\data\data\segmentation\ImageCAS_CT\ImageCAS-STACOM2025-02-10-2025\segmentations"
OUT_PATH = PROJECT_ROOT / "data/manifests_local/imagecas_1_200.csv"

if __name__ == "__main__":
    df = build_imagecas_manifest(IMAGE_ROOT, SEG_ROOT, max_slices_per_patient=10)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"imagecas_1_200: rows={len(df)} patients={df['patient_id'].nunique()}")
    print("wrote", OUT_PATH)
