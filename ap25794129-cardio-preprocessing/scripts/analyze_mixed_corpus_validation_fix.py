"""One-off comparison of the mixed-corpus validation-fix rerun's primary 10-epoch grid against
raw input, using the same Holm+bootstrap+TOST methodology as the rest of the manuscript."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cardiac_image_system.core.stats import compare_modes_to_baseline

MODES = ["none", "gaussian", "wavelet", "nlm", "clahe", "hybrid"]
BASE = Path("outputs/mixed_corpus_validation_fix_rerun/primary_grid")


def main() -> None:
    dice_by_mode = {}
    for mode in MODES:
        df = pd.read_csv(BASE / f"unet_combined_{mode}" / "test_patient_level.csv")
        dice_by_mode[mode] = df.set_index("patient_id")["dice"]

    result = compare_modes_to_baseline(dice_by_mode, baseline_mode="none", equivalence_margin=0.01)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print(result.to_string(index=False))
    out_path = Path("outputs/mixed_corpus_validation_fix_rerun_primary_grid_stats.csv")
    result.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
