import pytest

from joig_cardio.metrics import binary_metrics


def test_binary_metrics():
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.8, 0.9, 0.2])
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["specificity"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (1, 1, 1, 1)
