from __future__ import annotations

import numpy as np
import pytest

from neureptrace.metrics.weighted import (
    weighted_brier_score_multiclass,
    weighted_expected_calibration_error,
    weighted_negative_log_likelihood,
    weighted_top_k_accuracy,
)


def test_weighted_probability_metrics_clip_tiny_negative_roundoff() -> None:
    probabilities = np.array([[-1e-12, 1.0 + 1e-12], [0.3, 0.7]])
    labels = np.array([1, 1])
    sample_weight = np.array([2.0, 1.0])

    assert weighted_top_k_accuracy(probabilities, labels, sample_weight) == pytest.approx(1.0)
    assert weighted_negative_log_likelihood(probabilities, labels, sample_weight) == pytest.approx(
        -np.average(np.log([1.0, 0.7]), weights=sample_weight)
    )
    assert np.isfinite(weighted_brier_score_multiclass(probabilities, labels, sample_weight))
    assert np.isfinite(weighted_expected_calibration_error(probabilities, labels, sample_weight))


def test_weighted_probability_metrics_reject_substantive_negative_probabilities() -> None:
    probabilities = np.array([[-1e-3, 1.001], [0.3, 0.7]])
    labels = np.array([1, 1])
    sample_weight = np.array([1.0, 1.0])

    with pytest.raises(ValueError, match="non-negative"):
        weighted_top_k_accuracy(probabilities, labels, sample_weight)


def test_weighted_probability_metrics_reject_overflowing_row_sums_as_value_error() -> None:
    probabilities = np.array([[1e308, 1e308]])
    labels = np.array([0])
    sample_weight = np.array([1.0])

    with np.errstate(over="raise", invalid="raise", divide="raise"), pytest.raises(ValueError, match="sum to one"):
        weighted_top_k_accuracy(probabilities, labels, sample_weight)
