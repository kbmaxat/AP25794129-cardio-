"""Linear mixed-effects analysis of the 10-epoch, five-seed multiseed axis.

Fits, per dataset group (ACDC, CAMUS, ACDC+CAMUS), dice ~ mode with three
independent variance components: one per seed, one per mode x seed
combination (so the mode contrast itself, not only a shared seed-level
intercept, is allowed to vary by seed -- a diagonal-covariance equivalent of
a random slope for mode within seed), and one per patient.

The patient component is seed-nested for ACDC (patient_key = seed_patientid)
because the ACDC-only benchmark group reseeds its patient-level
train/val/test split with each run's training seed -- its native metadata
has no validation partition, so resolve_split_map falls back to
make_patient_level_random_split(seed=config.seed); see
train_unet_baseline.py. For CAMUS and the mixed corpus, which use CAMUS's
fixed official partition, patient identity is preserved across seeds, so the
patient component uses the raw patient_id and is fit as a genuinely crossed
random effect (not nested within seed) -- nesting it would treat repeated
measurements on the same patient across seeds as independent, which is
conservative for a difference test but anti-conservative for the
equivalence claims this analysis supports.

All variance components are fit as crossed effects (not nested within an
outer `groups` partition) using the standard statsmodels trick of a single
dummy `groups` value spanning the whole dataset, with every random-effects
term expressed through vc_formula.

An earlier version of this analysis used only a random intercept for seed
plus a nested random intercept for patient; that specification under-
propagated run-to-run uncertainty into the mode contrast itself. This
version is a direct correction, not an alternative -- see the manuscript's
mixed-effects subsection and README changelog for the full account.

Input: outputs/multiseed_long_format_all.csv (long-format, one row per
group/mode/seed/patient/dice; built by the multiseed data-gathering step
from the unet_binary_multiseed_* run trees).

Outputs:
  outputs/mixedlm_crossed_mode_effects.csv (mode coefficient, 95% CI,
    Holm-adjusted p-value per group)
  outputs/mixedlm_variance_components.csv (seed / mode:seed / patient /
    residual variance per group)
  outputs/mixedlm_seed_paired_sanity.csv (transparent, low-power paired
    Wilcoxon test on the five per-seed mean Dice values per mode, as a
    cross-check that does not depend on the mixed model's random-effects
    assumptions)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = REPO_ROOT / "outputs" / "multiseed_long_format_all.csv"
OUTPUT_DIR = REPO_ROOT / "outputs"

MODE_ORDER = ["none", "gaussian", "wavelet", "nlm", "clahe", "hybrid"]
NON_BASELINE_MODES = ["gaussian", "wavelet", "nlm", "clahe", "hybrid"]
GROUPS = ["ACDC", "CAMUS", "Combined"]
SEED_NESTED_PATIENT_GROUPS = {"ACDC"}


def fit_group(sub: pd.DataFrame, group: str) -> tuple[pd.DataFrame, dict[str, float]]:
    patient_term = "C(patient_key)" if group in SEED_NESTED_PATIENT_GROUPS else "C(patient_id)"
    vc_formula = {
        "seed": "0 + C(seed)",
        "mode_seed": "0 + C(mode):C(seed)",
        "patient": f"0 + {patient_term}",
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
    rows = []
    for name in best_fit.params.index:
        if not name.startswith("C(mode"):
            continue
        mode = name.split("T.")[1].rstrip("]")
        rows.append(
            {
                "mode": mode,
                "coef": best_fit.params[name],
                "se": best_fit.bse[name],
                "p": best_fit.pvalues[name],
                "ci_low": ci.loc[name, 0],
                "ci_high": ci.loc[name, 1],
                "converged": best_fit.converged,
            }
        )

    variance = dict(zip(model.exog_vc.names, best_fit.vcomp))
    variance["residual"] = best_fit.scale
    return pd.DataFrame(rows), variance


def main() -> None:
    full = pd.read_csv(INPUT_CSV)
    full["mode"] = pd.Categorical(full["mode"], categories=MODE_ORDER)
    full["patient_key"] = full["seed"].astype(str) + "_" + full["patient_id"].astype(str)
    full["dummy_group"] = 1

    all_rows = []
    vc_rows = []
    for group in GROUPS:
        sub = full[full["group"] == group].copy()
        group_result, variance = fit_group(sub, group)
        group_result.insert(0, "group", group)
        all_rows.append(group_result)
        for component, value in variance.items():
            vc_rows.append({"group": group, "component": component, "variance": value})

    result = pd.concat(all_rows, ignore_index=True)
    result["holm_p"] = None
    for group in result["group"].unique():
        mask = result["group"] == group
        _, p_adj, _, _ = multipletests(result.loc[mask, "p"], method="holm")
        result.loc[mask, "holm_p"] = p_adj

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_DIR / "mixedlm_crossed_mode_effects.csv", index=False)
    print(result.to_string(index=False))

    vc_df = pd.DataFrame(vc_rows)
    vc_df.to_csv(OUTPUT_DIR / "mixedlm_variance_components.csv", index=False)
    print()
    print(vc_df.to_string(index=False))

    sanity_rows = []
    for group in GROUPS:
        sub = full[full["group"] == group]
        seed_means = sub.groupby(["mode", "seed"], observed=True)["dice"].mean().unstack("mode")
        for mode in NON_BASELINE_MODES:
            diffs = (seed_means[mode] - seed_means["none"]).values
            _, p = wilcoxon(diffs)
            sanity_rows.append(
                {
                    "group": group,
                    "mode": mode,
                    "mean_diff_of_seed_means": diffs.mean(),
                    "n_seeds": len(diffs),
                    "wilcoxon_p": p,
                }
            )
    sanity_df = pd.DataFrame(sanity_rows)
    sanity_df.to_csv(OUTPUT_DIR / "mixedlm_seed_paired_sanity.csv", index=False)
    print()
    print(sanity_df.to_string(index=False))

    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
