from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

REQUIRED_COLUMNS = {"path", "label", "patient_id", "modality", "dataset"}
VALID_MODALITIES = {"echo", "mri"}


def load_manifest(path: str | Path, check_files: bool = True) -> pd.DataFrame:
    manifest_path = Path(path).resolve()
    frame = pd.read_csv(manifest_path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Manifest is empty")
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must be binary integers: 0 or 1")
    frame["modality"] = frame["modality"].str.lower()
    invalid = set(frame["modality"].unique()).difference(VALID_MODALITIES)
    if invalid:
        raise ValueError(f"Unsupported modalities: {sorted(invalid)}")
    base = manifest_path.parent
    frame["resolved_path"] = frame["path"].map(
        lambda value: str((base / value).resolve()) if not Path(value).is_absolute() else str(Path(value))
    )
    if check_files:
        absent = [value for value in frame["resolved_path"] if not Path(value).is_file()]
        if absent:
            preview = "\n".join(absent[:5])
            raise FileNotFoundError(f"{len(absent)} image files are missing. First entries:\n{preview}")
    return frame


def assign_grouped_folds(frame: pd.DataFrame, folds: int = 5, seed: int = 42) -> pd.DataFrame:
    if frame["patient_id"].nunique() < folds:
        raise ValueError(f"At least {folds} unique patients are required")
    result = frame.copy()
    result["fold"] = -1
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (_, validation_indices) in enumerate(
        splitter.split(result, result["label"], groups=result["patient_id"])
    ):
        result.loc[result.index[validation_indices], "fold"] = fold
    return result


class CardiacImageDataset:
    def __init__(self, frame: pd.DataFrame, transform=None):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = Image.open(row["resolved_path"]).convert("L")
        if self.transform:
            image = self.transform(image, row["modality"])
        return image, int(row["label"]), {
            "patient_id": str(row["patient_id"]),
            "path": str(row["resolved_path"]),
            "modality": str(row["modality"]),
            "dataset": str(row["dataset"]),
        }
