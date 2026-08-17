"""Full statistical reprocessing of every mixed-corpus branch affected by the validation-split
bug fix (see splits.py / test_carving_validation_carves_only_the_dataset_missing_native_validation
and scripts/run_mixed_corpus_validation_fix_rerun.py), using the manuscript's own established
methodology (Holm-adjusted paired Wilcoxon, percentile bootstrap CI, TOST equivalence at the
existing +/-0.01 Dice margin) so the results are directly comparable to every other table in the
manuscript.

Writes one CSV per branch into outputs/, plus prints a summary table to stdout.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cardiac_image_system.core.stats import compare_modes_to_baseline

MODES = ["none", "gaussian", "wavelet", "nlm", "clahe", "hybrid"]
TOP3_MODES = ["none", "wavelet", "nlm"]
SEEDS = [11, 22, 33, 44, 55]
ROOT = Path("outputs/mixed_corpus_validation_fix_rerun")
OUT = Path("outputs")


def load_dice(path: Path) -> pd.Series:
    df = pd.read_csv(path / "test_patient_level.csv")
    return df.set_index("patient_id")["dice"]


def run_and_save(dice_by_mode: dict[str, pd.Series], label: str) -> pd.DataFrame:
    result = compare_modes_to_baseline(dice_by_mode, baseline_mode="none", equivalence_margin=0.01)
    result.insert(0, "branch", label)
    return result


def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    all_results = []

    # 1. Long-schedule
    dice_by_mode = {m: load_dice(ROOT / "longschedule" / f"unet_combined_{m}") for m in MODES}
    all_results.append(run_and_save(dice_by_mode, "longschedule"))

    # 2. Attention U-Net (top-3 modes, primary budget)
    dice_by_mode = {m: load_dice(ROOT / "attention_unet" / f"attn_unet_combined_{m}") for m in TOP3_MODES}
    all_results.append(run_and_save(dice_by_mode, "attention_unet"))

    # 3. TransUNet (top-3 modes, primary budget)
    dice_by_mode = {m: load_dice(ROOT / "transunet" / f"transunet_combined_{m}") for m in TOP3_MODES}
    all_results.append(run_and_save(dice_by_mode, "transunet"))

    combined = pd.concat(all_results, ignore_index=True)
    print(combined.to_string(index=False))
    out_path = OUT / "mixed_corpus_validation_fix_rerun_full_stats.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # 4. Multiseed raw means (five-seed axis), for direct before/after comparison against
    #    the manuscript's existing Table 7 (raw five-seed means) -- full mixed-effects refit is
    #    a separate follow-up step, this just reports the descriptive seed-mean shift first.
    print("\n=== Multiseed (10-epoch, 5-seed) patient-level mean Dice per mode ===")
    seed_means = {}
    for mode in MODES:
        vals = []
        for seed in SEEDS:
            d = load_dice(ROOT / "multiseed" / f"unet_combined_{mode}_seed{seed}")
            vals.append(d.mean())
        seed_means[mode] = sum(vals) / len(vals)
        print(f"{mode}: mean-of-seed-means dice = {seed_means[mode]:.4f} (seeds: {[round(v, 4) for v in vals]})")


if __name__ == "__main__":
    main()
