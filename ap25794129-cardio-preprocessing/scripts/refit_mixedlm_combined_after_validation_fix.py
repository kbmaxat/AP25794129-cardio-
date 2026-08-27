"""Refit the mixed-effects model for the 'Combined' (mixed-corpus) dataset group only, using
the corrected multiseed data from the validation-split bug fix (see splits.py and
scripts/run_mixed_corpus_validation_fix_rerun.py). ACDC and CAMUS multiseed data are untouched by
that bug (single-dataset manifests already had complete native validation coverage) and are not
refit here -- their existing coefficients in outputs/mixedlm_crossed_mode_effects.csv remain
valid.

Builds a corrected long-format table (old 'Combined' rows replaced with the new rerun's
patient-level dice) as a new file, then fits the exact same model specification used in
analyze_multiseed_mixedlm.py's fit_group for the 'Combined' group, for direct before/after
comparison against the manuscript's existing Table 8 (mixed-effects coefficients) and Table 9
(variance-component decomposition).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

REPO_ROOT = Path(__file__).resolve().parents[1]
OLD_LONG_FORMAT = REPO_ROOT / "outputs" / "multiseed_long_format_all.csv"
RERUN_ROOT = REPO_ROOT / "outputs" / "mixed_corpus_validation_fix_rerun" / "multiseed"
OUTPUT_DIR = REPO_ROOT / "outputs"

MODES = ["none", "gaussian", "wavelet", "nlm", "clahe", "hybrid"]
SEEDS = [11, 22, 33, 44, 55]


def build_corrected_combined_rows() -> pd.DataFrame:
    rows = []
    for mode in MODES:
        for seed in SEEDS:
            df = pd.read_csv(RERUN_ROOT / f"unet_combined_{mode}_seed{seed}" / "test_patient_level.csv")
            for _, r in df.iterrows():
                rows.append({"group": "Combined", "mode": mode, "seed": seed,
                             "patient_id": r["patient_id"], "dice": r["dice"]})
    return pd.DataFrame(rows)


def fit_group(sub: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    vc_formula = {
        "seed": "0 + C(seed)",
        "mode_seed": "0 + C(mode):C(seed)",
        "patient": "0 + C(patient_id)",
    }
    model = smf.mixedlm(
        "dice ~ C(mode, Treatment(reference='none'))",
        data=sub,
        groups=sub["dummy_group"],
        vc_formula=vc_formula,
    )
    best_fit = None
    for method in ["bfgs", "powell", "lbfgs"]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = model.fit(reml=True, method=[method])
        if best_fit is None or fit.llf > best_fit.llf:
            best_fit = fit

    ci = best_fit.conf_int()
    out_rows = []
    for name in best_fit.params.index:
        if not name.startswith("C(mode"):
            continue
        mode = name.split("T.")[1].rstrip("]")
        out_rows.append({
            "mode": mode, "coef": best_fit.params[name], "se": best_fit.bse[name],
            "p": best_fit.pvalues[name], "ci_low": ci.loc[name, 0], "ci_high": ci.loc[name, 1],
            "converged": best_fit.converged,
        })
    variance = dict(zip(model.exog_vc.names, best_fit.vcomp))
    variance["residual"] = best_fit.scale
    return pd.DataFrame(out_rows), variance


def main() -> None:
    from statsmodels.stats.multitest import multipletests

    old = pd.read_csv(OLD_LONG_FORMAT)
    unaffected = old[old["group"] != "Combined"].copy()
    corrected_combined = build_corrected_combined_rows()
    corrected_full = pd.concat([unaffected, corrected_combined], ignore_index=True)
    corrected_full["mode"] = pd.Categorical(corrected_full["mode"], categories=MODES)
    corrected_full["dummy_group"] = 1
    out_long_path = OUTPUT_DIR / "multiseed_long_format_all_mixed_corpus_fixed.csv"
    corrected_full.to_csv(out_long_path, index=False)
    print(f"Corrected long-format table saved to {out_long_path}")

    sub = corrected_full[corrected_full["group"] == "Combined"].copy()
    print(f"\nRefitting 'Combined' group on corrected data (n={len(sub)} rows)...")
    result, variance = fit_group(sub)
    _, holm_p, _, _ = multipletests(result["p"], method="holm")
    result["holm_p"] = holm_p
    result.insert(0, "group", "Combined")

    pd.set_option("display.width", 160)
    print("\n=== Corrected mixed-effects coefficients, Combined group ===")
    print(result.to_string(index=False))
    result.to_csv(OUTPUT_DIR / "mixedlm_combined_refit_after_validation_fix.csv", index=False)

    print("\n=== Corrected variance components, Combined group ===")
    vc_df = pd.DataFrame([{"component": k, "variance": v} for k, v in variance.items()])
    print(vc_df.to_string(index=False))
    vc_df.to_csv(OUTPUT_DIR / "mixedlm_combined_refit_variance_after_validation_fix.csv", index=False)


if __name__ == "__main__":
    main()
