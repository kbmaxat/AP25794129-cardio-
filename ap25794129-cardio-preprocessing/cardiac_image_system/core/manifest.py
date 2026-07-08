from __future__ import annotations

from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {"patient_id", "phase", "image_path", "mask_path"}


def load_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    if df["patient_id"].isna().any():
        raise ValueError("patient_id contains missing values")
    return df


def summarize_manifest(df: pd.DataFrame) -> dict:
    summary = {
        "num_rows": int(len(df)),
        "num_patients": int(df["patient_id"].nunique()),
        "phases": sorted(map(str, df["phase"].dropna().unique())),
    }
    if "dataset" in df.columns:
        summary["datasets"] = sorted(map(str, df["dataset"].dropna().unique()))
    if "subset" in df.columns:
        summary["subsets"] = sorted(map(str, df["subset"].dropna().unique()))
    return summary
