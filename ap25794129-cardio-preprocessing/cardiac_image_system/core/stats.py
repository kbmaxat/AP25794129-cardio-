from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (matches R's p.adjust(method="holm"))."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p)
    sorted_p = p[order]

    adjusted_sorted = np.empty(m, dtype=float)
    running_max = 0.0
    for i in range(m):
        running_max = max(running_max, (m - i) * sorted_p[i])
        adjusted_sorted[i] = min(running_max, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def bootstrap_ci_mean_diff(
    differences: np.ndarray,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of paired differences."""
    diffs = np.asarray(differences, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    if diffs.size == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, diffs.size, size=(n_bootstrap, diffs.size))
    boot_means = diffs[sample_idx].mean(axis=1)

    alpha = 1.0 - ci_level
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lower, upper


def wilcoxon_signed_rank(differences: np.ndarray) -> tuple[float, float]:
    """Paired Wilcoxon signed-rank test on differences. Returns (statistic, p_value)."""
    diffs = np.asarray(differences, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    diffs = diffs[diffs != 0]
    if diffs.size == 0:
        return float("nan"), float("nan")
    statistic, p_value = scipy_stats.wilcoxon(diffs)
    return float(statistic), float(p_value)


def tost_paired(differences: np.ndarray, margin: float, alpha: float = 0.05) -> dict[str, float | bool]:
    """Two one-sided tests (TOST) for equivalence of paired differences within +/- margin.

    Complements a non-significant Wilcoxon/Holm result: failing to reject H0 (no difference)
    does not prove equivalence, whereas a significant TOST result does.
    """
    if margin <= 0:
        raise ValueError("margin must be positive")

    diffs = np.asarray(differences, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    n = diffs.size
    if n < 2:
        raise ValueError("tost_paired requires at least two paired observations")

    mean = float(diffs.mean())
    se = float(diffs.std(ddof=1) / np.sqrt(n))
    df = n - 1

    if se == 0.0:
        p_lower = 0.0 if mean > -margin else 1.0
        p_upper = 0.0 if mean < margin else 1.0
    else:
        t_lower = (mean - (-margin)) / se
        t_upper = (mean - margin) / se
        p_lower = float(1.0 - scipy_stats.t.cdf(t_lower, df))
        p_upper = float(scipy_stats.t.cdf(t_upper, df))

    p_tost = max(p_lower, p_upper)
    return {
        "mean_diff": mean,
        "margin": float(margin),
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_tost": p_tost,
        "equivalent": bool(p_tost < alpha),
    }


def compare_modes_to_baseline(
    patient_metric_by_mode: dict[str, pd.Series],
    baseline_mode: str,
    equivalence_margin: float = 0.01,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    alpha: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Paired mode-vs-baseline comparison table (mirrors docs/results/table3_dice_vs_none_inference.csv).

    ``patient_metric_by_mode`` maps a preprocessing mode name to a pandas Series of a single
    patient-level metric (e.g. Dice) indexed by ``patient_id``. Series are paired on the
    intersection of patient IDs so that only patients evaluated under both the baseline and
    the compared mode contribute to the difference.
    """
    if baseline_mode not in patient_metric_by_mode:
        raise ValueError(f"baseline_mode '{baseline_mode}' not found in patient_metric_by_mode")

    baseline = patient_metric_by_mode[baseline_mode]
    modes_order = [mode for mode in patient_metric_by_mode if mode != baseline_mode]
    if not modes_order:
        raise ValueError("patient_metric_by_mode must contain at least one non-baseline mode")

    diffs_by_mode: dict[str, np.ndarray] = {}
    raw_p_by_mode: dict[str, float] = {}
    n_by_mode: dict[str, int] = {}

    for mode in modes_order:
        paired = pd.concat(
            [baseline.rename("baseline"), patient_metric_by_mode[mode].rename("mode")],
            axis=1,
            join="inner",
        ).dropna()
        if paired.empty:
            raise ValueError(f"No overlapping patient IDs between baseline '{baseline_mode}' and mode '{mode}'")

        diffs = (paired["mode"] - paired["baseline"]).to_numpy()
        diffs_by_mode[mode] = diffs
        n_by_mode[mode] = int(diffs.size)
        _, wilcoxon_p = wilcoxon_signed_rank(diffs)
        raw_p_by_mode[mode] = wilcoxon_p

    holm_p_values = holm_bonferroni([raw_p_by_mode[mode] for mode in modes_order])

    rows = []
    for mode, holm_p in zip(modes_order, holm_p_values):
        diffs = diffs_by_mode[mode]
        ci_low, ci_high = bootstrap_ci_mean_diff(diffs, n_bootstrap=n_bootstrap, ci_level=ci_level, seed=seed)
        tost = tost_paired(diffs, margin=equivalence_margin, alpha=alpha)
        rows.append(
            {
                "mode": mode,
                "n_patients": n_by_mode[mode],
                "mean_diff": float(np.mean(diffs)),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "wilcoxon_p": raw_p_by_mode[mode],
                "holm_p": holm_p,
                "tost_p": tost["p_tost"],
                "tost_equivalent": tost["equivalent"],
            }
        )
    return pd.DataFrame(rows)
