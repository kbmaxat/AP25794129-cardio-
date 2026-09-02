"""CAMUS image-quality-stratified analysis of the primary six-mode benchmark.

CAMUS ships a per-view expert image-quality label (Good/Medium/Poor) in each
patient's Info_2CH.cfg / Info_4CH.cfg. This was not otherwise used in the
benchmark. Each CAMUS test patient is assigned the worse of its two available
view-quality labels, and the CAMUS long-schedule six-mode rerun's
patient-level Dice (up to 50 epochs, early stopping -- the same run reported
in the manuscript's full six-mode long-schedule confirmation table; top-3
modes from unet_binary_longschedule_camus_top3, remaining three modes from
phase2_full_longschedule/longschedule_camus_remaining3) is re-analyzed within
each quality tier using the same paired Holm-adjusted Wilcoxon test,
bootstrap confidence interval, and TOST equivalence procedure (margin
+/-0.01 Dice) used throughout this benchmark.

Outputs:
  outputs/camus_test_patient_quality.csv
  outputs/camus_quality_stratified_descriptive.csv
  outputs/camus_quality_stratified_stats.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cardiac_image_system.core.stats import compare_modes_to_baseline

REPO_ROOT = Path(__file__).resolve().parents[1]
CAMUS_NIFTI_ROOT = Path(
    r"D:\article\Обзорные статьи\data\segmentation\CAMUS_full\database_nifti"
)
LONGSCHEDULE_TOP3_RUN = (
    REPO_ROOT / "outputs" / "unet_binary_longschedule_camus_top3" / "20260703_104256"
)
LONGSCHEDULE_REMAINING3_RUN = (
    REPO_ROOT / "outputs" / "phase2_full_longschedule" / "longschedule_camus_remaining3"
)
TOP3_MODES = ["none", "wavelet", "nlm"]
REMAINING3_MODES = ["gaussian", "clahe", "hybrid"]
MODES = ["none", "gaussian", "wavelet", "nlm", "clahe", "hybrid"]


def mode_run_dir(mode: str) -> Path:
    if mode in TOP3_MODES:
        return LONGSCHEDULE_TOP3_RUN / f"unet_camus_{mode}"
    return LONGSCHEDULE_REMAINING3_RUN / f"unet_camus_{mode}"
QUALITY_ORDER = ["Poor", "Medium", "Good"]

OUTPUT_DIR = REPO_ROOT / "outputs"


def read_quality(cfg_path: Path) -> str | None:
    if not cfg_path.exists():
        return None
    for line in cfg_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("ImageQuality:"):
            return line.split(":", 1)[1].strip()
    return None


def worst_quality(labels: list[str]) -> str:
    rank = {"Poor": 0, "Medium": 1, "Good": 2}
    return min(labels, key=lambda label: rank[label])


def build_patient_quality_table(patient_ids: list[str]) -> pd.DataFrame:
    rows = []
    for full_id in patient_ids:
        patient_dir_name = full_id.replace("CAMUS_", "")
        patient_dir = CAMUS_NIFTI_ROOT / patient_dir_name
        labels = []
        for view in ["2CH", "4CH"]:
            quality = read_quality(patient_dir / f"Info_{view}.cfg")
            if quality is not None:
                labels.append(quality)
        if not labels:
            raise ValueError(f"No quality metadata found for {full_id}")
        rows.append({"patient_id": full_id, "quality": worst_quality(labels)})
    return pd.DataFrame(rows)


def main() -> None:
    none_patients = pd.read_csv(mode_run_dir("none") / "test_patient_level.csv")
    patient_ids = sorted(none_patients["patient_id"].unique().tolist())

    quality_table = build_patient_quality_table(patient_ids)
    quality_table.to_csv(OUTPUT_DIR / "camus_test_patient_quality.csv", index=False)
    print(quality_table["quality"].value_counts())

    per_mode = {}
    for mode in MODES:
        df = pd.read_csv(mode_run_dir(mode) / "test_patient_level.csv")
        per_mode[mode] = df.merge(quality_table, on="patient_id", how="inner")

    descriptive_rows = []
    for quality in QUALITY_ORDER:
        for mode in MODES:
            sub = per_mode[mode][per_mode[mode]["quality"] == quality]
            descriptive_rows.append(
                {
                    "quality": quality,
                    "mode": mode,
                    "n": len(sub),
                    "mean_dice": sub["dice"].mean(),
                    "sd": sub["dice"].std(ddof=1),
                }
            )
    descriptive = pd.DataFrame(descriptive_rows)
    descriptive.to_csv(OUTPUT_DIR / "camus_quality_stratified_descriptive.csv", index=False)

    stats_frames = []
    for quality in QUALITY_ORDER:
        patient_metric_by_mode = {
            mode: per_mode[mode][per_mode[mode]["quality"] == quality].set_index("patient_id")["dice"]
            for mode in MODES
        }
        result = compare_modes_to_baseline(patient_metric_by_mode, baseline_mode="none", equivalence_margin=0.01)
        result.insert(0, "quality", quality)
        stats_frames.append(result)
    stats = pd.concat(stats_frames, ignore_index=True)
    stats.to_csv(OUTPUT_DIR / "camus_quality_stratified_stats.csv", index=False)

    print(descriptive.to_string(index=False))
    print(stats.to_string(index=False))
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
