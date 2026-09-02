from __future__ import annotations

from pathlib import Path
import re

import nibabel as nib
import numpy as np
import pandas as pd

from cardiac_image_system.core.io import get_nifti_in_plane_spacing

FRAME_PATTERN = re.compile(r"_frame(\d+)\.nii(?:\.gz)?$", re.IGNORECASE)


def parse_acdc_info_cfg(path: str | Path) -> dict[str, str | int | float]:
    path = Path(path)
    info: dict[str, str | int | float] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if re.fullmatch(r"-?\d+", value):
                info[key] = int(value)
            else:
                try:
                    info[key] = float(value)
                except ValueError:
                    info[key] = value
    return info


def _strip_nii_suffix(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".nii.gz"):
        return name[:-7]
    if lower.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def _phase_from_frame(frame_id: int, info: dict[str, str | int | float]) -> str:
    ed = info.get("ED")
    es = info.get("ES")
    if ed is not None and int(ed) == frame_id:
        return "diastole"
    if es is not None and int(es) == frame_id:
        return "systole"
    return f"frame_{frame_id:02d}"


def build_acdc_manifest(
    dataset_root: str | Path,
    subsets: tuple[str, ...] = ("training", "testing"),
    include_empty_masks: bool = False,
) -> pd.DataFrame:
    dataset_root = Path(dataset_root)
    rows: list[dict] = []

    for subset in subsets:
        subset_dir = dataset_root / subset
        if not subset_dir.exists():
            continue
        for patient_dir in sorted(p for p in subset_dir.iterdir() if p.is_dir()):
            info_path = patient_dir / "Info.cfg"
            if not info_path.exists():
                continue
            info = parse_acdc_info_cfg(info_path)
            source_patient_id = patient_dir.name
            patient_id = f"ACDC_{source_patient_id}"

            image_paths = sorted(
                p for p in patient_dir.glob("*_frame*.nii.gz") if not p.name.endswith("_gt.nii.gz")
            )
            for image_path in image_paths:
                match = FRAME_PATTERN.search(image_path.name)
                if not match:
                    continue
                frame_id = int(match.group(1))
                image_stem = _strip_nii_suffix(image_path.name)
                mask_path = image_path.parent / f"{image_stem}_gt.nii.gz"
                if not mask_path.exists():
                    continue

                mask_array = nib.load(mask_path).get_fdata(dtype=np.float32)
                if mask_array.ndim == 2:
                    mask_slices = [mask_array]
                elif mask_array.ndim == 3:
                    mask_slices = [mask_array[:, :, i] for i in range(mask_array.shape[2])]
                else:
                    raise ValueError(f"Unsupported ACDC mask dimensionality: {mask_array.shape} at {mask_path}")

                spacing = get_nifti_in_plane_spacing(image_path)
                spacing_row_mm = spacing[0] if spacing else None
                spacing_col_mm = spacing[1] if spacing else None

                phase = _phase_from_frame(frame_id, info)
                for slice_index, mask_slice in enumerate(mask_slices):
                    has_positive = bool(np.any(mask_slice > 0))
                    if not include_empty_masks and not has_positive:
                        continue
                    rows.append(
                        {
                            "patient_id": patient_id,
                            "source_patient_id": source_patient_id,
                            "phase": phase,
                            "image_path": str(image_path),
                            "mask_path": str(mask_path),
                            "slice_index": slice_index,
                            "dataset": "ACDC",
                            "subset": subset,
                            "modality": "cine_cmr",
                            "view": "SAX",
                            "frame_id": frame_id,
                            "group": info.get("Group", "unknown"),
                            "has_positive_mask": has_positive,
                            "spacing_row_mm": spacing_row_mm,
                            "spacing_col_mm": spacing_col_mm,
                        }
                    )

    df = pd.DataFrame(rows)
    sort_cols = [col for col in ["dataset", "subset", "patient_id", "frame_id", "slice_index"] if col in df.columns]
    if not df.empty and sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def _load_patient_split_map(split_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    file_to_subset = {
        "subgroup_training.txt": "training",
        "subgroup_validation.txt": "validation",
        "subgroup_testing.txt": "testing",
    }
    for file_name, subset in file_to_subset.items():
        split_file = split_dir / file_name
        if not split_file.exists():
            continue
        with split_file.open("r", encoding="utf-8") as f:
            for raw_line in f:
                patient_id = raw_line.strip()
                if patient_id:
                    mapping[patient_id] = subset
    return mapping


def build_camus_manifest(
    dataset_root: str | Path,
    include_half_sequences: bool = False,
) -> pd.DataFrame:
    dataset_root = Path(dataset_root)
    nifti_root = dataset_root / "database_nifti"
    split_root = dataset_root / "database_split"
    split_map = _load_patient_split_map(split_root) if split_root.exists() else {}
    rows: list[dict] = []

    for patient_dir in sorted(p for p in nifti_root.iterdir() if p.is_dir()):
        source_patient_id = patient_dir.name
        patient_id = f"CAMUS_{source_patient_id}"
        subset = split_map.get(source_patient_id, "unassigned")

        for view in ("2CH", "4CH"):
            for phase_code, phase_name in (("ED", "diastole"), ("ES", "systole")):
                image_path = patient_dir / f"{source_patient_id}_{view}_{phase_code}.nii.gz"
                mask_path = patient_dir / f"{source_patient_id}_{view}_{phase_code}_gt.nii.gz"
                if image_path.exists() and mask_path.exists():
                    spacing = get_nifti_in_plane_spacing(image_path)
                    rows.append(
                        {
                            "patient_id": patient_id,
                            "source_patient_id": source_patient_id,
                            "phase": phase_name,
                            "image_path": str(image_path),
                            "mask_path": str(mask_path),
                            "slice_index": 0,
                            "dataset": "CAMUS",
                            "subset": subset,
                            "modality": "echocardiography",
                            "view": view,
                            "frame_id": phase_code,
                            "group": "public",
                            "has_positive_mask": True,
                            "spacing_row_mm": spacing[0] if spacing else None,
                            "spacing_col_mm": spacing[1] if spacing else None,
                        }
                    )

            if include_half_sequences:
                image_path = patient_dir / f"{source_patient_id}_{view}_half_sequence.nii.gz"
                mask_path = patient_dir / f"{source_patient_id}_{view}_half_sequence_gt.nii.gz"
                if image_path.exists() and mask_path.exists():
                    spacing = get_nifti_in_plane_spacing(image_path)
                    rows.append(
                        {
                            "patient_id": patient_id,
                            "source_patient_id": source_patient_id,
                            "phase": "half_sequence",
                            "image_path": str(image_path),
                            "mask_path": str(mask_path),
                            "slice_index": 0,
                            "dataset": "CAMUS",
                            "subset": subset,
                            "modality": "echocardiography",
                            "view": view,
                            "frame_id": "half_sequence",
                            "group": "public",
                            "has_positive_mask": True,
                            "spacing_row_mm": spacing[0] if spacing else None,
                            "spacing_col_mm": spacing[1] if spacing else None,
                        }
                    )

    df = pd.DataFrame(rows)
    sort_cols = [col for col in ["dataset", "subset", "patient_id", "view", "phase"] if col in df.columns]
    if not df.empty and sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def build_imagecas_manifest(
    image_root: str | Path,
    segmentation_root: str | Path,
    max_slices_per_patient: int = 10,
) -> pd.DataFrame:
    """Build a manifest for the ImageCAS cardiac CTA dataset with the STACOM 2025 whole-heart
    labelmaps (Hansen et al., 2025), which relabel the original coronary-only annotations into
    11 structures (background, myocardium, LA, LV, RA, RV, aorta, PA, LAA, coronary, PV).

    Both ``image_root`` (per-patient ``{id}.img.nii.gz`` CT volumes) and ``segmentation_root``
    (per-patient ``{id}.img.nii.gz`` labelmaps, same filename convention as the images) are
    required, since the raw ImageCAS images and the STACOM labelmaps are distributed separately
    and matched only by shared numeric patient id.

    A CT volume can have 200+ axial slices with a non-empty label, far more than the handful of
    cine-MRI phases per ACDC patient; to keep per-patient contribution comparable to the other
    datasets in this benchmark (and avoid one CT patient dominating a mixed-corpus batch), at
    most ``max_slices_per_patient`` positive-mask slices are kept per patient, evenly spaced
    across that patient's positive-mask z-range rather than the first N encountered.
    """
    image_root = Path(image_root)
    segmentation_root = Path(segmentation_root)
    rows: list[dict] = []

    image_paths = sorted(
        image_root.glob("*.img.nii.gz"),
        key=lambda p: int(p.name.split(".")[0]) if p.name.split(".")[0].isdigit() else p.name,
    )
    for image_path in image_paths:
        source_patient_id = image_path.name.split(".")[0]
        mask_path = segmentation_root / image_path.name
        if not mask_path.exists():
            continue

        mask_img = nib.load(mask_path)
        mask_array = mask_img.get_fdata(dtype=np.float32)
        if mask_array.ndim != 3:
            raise ValueError(f"Unsupported ImageCAS mask dimensionality: {mask_array.shape} at {mask_path}")

        positive_slices = [z for z in range(mask_array.shape[2]) if np.any(mask_array[:, :, z] > 0)]
        if not positive_slices:
            continue
        if len(positive_slices) > max_slices_per_patient:
            pick_idx = np.linspace(0, len(positive_slices) - 1, num=max_slices_per_patient)
            positive_slices = sorted({positive_slices[int(round(i))] for i in pick_idx})

        spacing = get_nifti_in_plane_spacing(image_path)
        patient_id = f"ImageCAS_{source_patient_id}"
        for slice_index in positive_slices:
            rows.append(
                {
                    "patient_id": patient_id,
                    "source_patient_id": source_patient_id,
                    "phase": "static",
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "slice_index": slice_index,
                    "dataset": "ImageCAS",
                    "subset": "unassigned",
                    "modality": "cardiac_ct",
                    "view": "axial",
                    "frame_id": 0,
                    "group": "public",
                    "has_positive_mask": True,
                    "spacing_row_mm": spacing[0] if spacing else None,
                    "spacing_col_mm": spacing[1] if spacing else None,
                }
            )

    df = pd.DataFrame(rows)
    sort_cols = [col for col in ["dataset", "patient_id", "slice_index"] if col in df.columns]
    if not df.empty and sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def summarize_manifest_by_group(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["dataset", "subset", "rows", "patients"])
    group_cols = [col for col in ["dataset", "subset"] if col in df.columns]
    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(rows=("patient_id", "size"), patients=("patient_id", "nunique"))
        .reset_index()
    )
    return grouped
