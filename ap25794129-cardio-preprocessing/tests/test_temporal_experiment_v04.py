from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "temporal_v01"))

from run_temporal_information_check import crossed_bootstrap


def test_crossed_bootstrap_is_exact_for_constant_effect():
    values = np.full((6, 80), 0.25)
    interval = crossed_bootstrap(values, seed=7, replicates=100)
    assert interval == pytest.approx([0.25, 0.25])


def test_crossed_bootstrap_is_reproducible():
    values = np.arange(24, dtype=np.float64).reshape(4, 6)
    first = crossed_bootstrap(values, seed=2027, replicates=500)
    second = crossed_bootstrap(values, seed=2027, replicates=500)
    assert first == second


def test_crossed_bootstrap_rejects_single_cluster_dimension():
    with pytest.raises(ValueError):
        crossed_bootstrap(np.ones((1, 5)), seed=0)
    with pytest.raises(ValueError):
        crossed_bootstrap(np.ones((5, 1)), seed=0)
