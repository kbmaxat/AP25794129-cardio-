import numpy as np
import pandas as pd
import pytest

from cardiac_image_system.core.stats import (
    bootstrap_ci_mean_diff,
    compare_modes_to_baseline,
    holm_bonferroni,
    tost_paired,
    wilcoxon_signed_rank,
)


def test_holm_bonferroni_matches_known_reference_values():
    # Reference values cross-checked against R's p.adjust(c(0.01, 0.04, 0.03, 0.005), method="holm").
    p_values = [0.01, 0.04, 0.03, 0.005]
    adjusted = holm_bonferroni(p_values)
    assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_bonferroni_is_never_less_strict_than_raw_p():
    p_values = [0.001, 0.02, 0.2, 0.5]
    adjusted = holm_bonferroni(p_values)
    for raw, adj in zip(p_values, adjusted):
        assert adj >= raw


def test_bootstrap_ci_excludes_zero_for_clear_effect():
    rng = np.random.default_rng(0)
    differences = rng.normal(loc=0.05, scale=0.01, size=200)
    low, high = bootstrap_ci_mean_diff(differences, n_bootstrap=2000, seed=1)
    assert low > 0.0
    assert low < high


def test_bootstrap_ci_contains_zero_for_no_effect():
    rng = np.random.default_rng(0)
    differences = rng.normal(loc=0.0, scale=0.05, size=200)
    low, high = bootstrap_ci_mean_diff(differences, n_bootstrap=2000, seed=1)
    assert low < 0.0 < high


def test_wilcoxon_signed_rank_significant_for_consistent_shift():
    rng = np.random.default_rng(0)
    differences = rng.normal(loc=0.1, scale=0.02, size=50)
    _, p_value = wilcoxon_signed_rank(differences)
    assert p_value < 0.05


def test_wilcoxon_signed_rank_handles_all_zero_gracefully():
    statistic, p_value = wilcoxon_signed_rank(np.zeros(10))
    assert np.isnan(statistic)
    assert np.isnan(p_value)


def test_tost_paired_declares_equivalence_for_tiny_differences():
    rng = np.random.default_rng(0)
    differences = rng.normal(loc=0.0, scale=0.002, size=100)
    result = tost_paired(differences, margin=0.01)
    assert result["equivalent"] is True


def test_tost_paired_rejects_equivalence_for_large_difference():
    rng = np.random.default_rng(0)
    differences = rng.normal(loc=0.05, scale=0.01, size=100)
    result = tost_paired(differences, margin=0.01)
    assert result["equivalent"] is False


def test_tost_paired_requires_positive_margin():
    with pytest.raises(ValueError):
        tost_paired(np.array([0.0, 0.01]), margin=0.0)


def test_compare_modes_to_baseline_reproduces_paired_mean_difference():
    baseline = pd.Series({"p1": 0.90, "p2": 0.85, "p3": 0.88}, name="none")
    wavelet = pd.Series({"p1": 0.91, "p2": 0.87, "p3": 0.89}, name="wavelet")
    clahe = pd.Series({"p1": 0.80, "p2": 0.75, "p3": 0.78}, name="clahe")

    result = compare_modes_to_baseline(
        {"none": baseline, "wavelet": wavelet, "clahe": clahe},
        baseline_mode="none",
        n_bootstrap=500,
        seed=1,
    )

    result = result.set_index("mode")
    assert result.loc["wavelet", "n_patients"] == 3
    assert result.loc["wavelet", "mean_diff"] == pytest.approx((0.01 + 0.02 + 0.01) / 3)
    # clahe is clearly worse than baseline on every patient
    assert result.loc["clahe", "mean_diff"] < result.loc["wavelet", "mean_diff"]
    # Holm correction must never make a p-value smaller than the raw Wilcoxon p-value
    assert (result["holm_p"] >= result["wilcoxon_p"]).all()


def test_compare_modes_to_baseline_pairs_on_intersection_only():
    baseline = pd.Series({"p1": 0.90, "p2": 0.85, "p3": 0.80}, name="none")
    partial = pd.Series({"p1": 0.92, "p2": 0.87, "p4": 0.99}, name="partial_mode")

    result = compare_modes_to_baseline(
        {"none": baseline, "partial_mode": partial},
        baseline_mode="none",
        n_bootstrap=200,
        seed=1,
    )
    # p4 has no baseline value and p3 has no partial_mode value, so only p1/p2 are paired.
    assert result.loc[0, "n_patients"] == 2
    assert result.loc[0, "mean_diff"] == pytest.approx((0.02 + 0.02) / 2)


def test_compare_modes_to_baseline_raises_on_single_overlapping_patient():
    baseline = pd.Series({"p1": 0.90, "p2": 0.85}, name="none")
    partial = pd.Series({"p1": 0.92, "p3": 0.99}, name="partial_mode")
    with pytest.raises(ValueError):
        compare_modes_to_baseline(
            {"none": baseline, "partial_mode": partial},
            baseline_mode="none",
            n_bootstrap=200,
            seed=1,
        )


def test_compare_modes_to_baseline_requires_known_baseline():
    baseline = pd.Series({"p1": 0.9}, name="none")
    with pytest.raises(ValueError):
        compare_modes_to_baseline({"none": baseline}, baseline_mode="missing")
