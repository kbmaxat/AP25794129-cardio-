from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "temporal_v01"))

from summarize_correction_penalty_ablation import crossed_bootstrap


def test_crossed_bootstrap_is_exact_for_constant_effect():
    interval = crossed_bootstrap(np.full((6, 80), -0.25), seed=17, replicates=200)
    assert interval == pytest.approx([-0.25, -0.25])


def test_crossed_bootstrap_is_reproducible():
    values = np.arange(24, dtype=np.float64).reshape(4, 6)
    assert crossed_bootstrap(values, seed=2027, replicates=500) == crossed_bootstrap(
        values, seed=2027, replicates=500
    )


@pytest.mark.parametrize("values", [np.ones((1, 5)), np.ones((5, 1))])
def test_crossed_bootstrap_requires_patient_and_seed_clusters(values):
    with pytest.raises(ValueError):
        crossed_bootstrap(values, seed=0)
