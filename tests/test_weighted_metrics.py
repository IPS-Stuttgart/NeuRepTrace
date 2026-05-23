from __future__ import annotations

import numpy as np
import pytest

from neureptrace.metrics.weighted import (
    validate_sample_weight,
    weighted_brier_score_multiclass,
    weighted_expected_calibration_error,
    weighted_negative_log_likelihood,
    weighted_top_k_accuracy,
)


def test_weighted_probability_metrics_match_manual_averages() -> None:
    probabilities = np.array(
        [
            [0.9, 0.1],
            [0.4, 0.6],
            [0.8, 0.2],
        ]
    )
    labels = np.array([0, 1, 1])
    sample_weight = np.array([1.0, 2.0, 3.0])

    assert weighted_brier_score_multiclass(probabilities, labels, sample_weight) == pytest.approx(0.75)
    assert weighted_negative_log_likelihood(probabilities, labels, sample_weight) == pytest.approx(
        -np.average(np.log([0.9, 0.6, 0.2]), weights=sample_weight)
    )
    assert weighted_top_k_accuracy(probabilities, labels, sample_weight, k=1) == pytest.approx(0.5)
    assert weighted_top_k_accuracy(probabilities, labels, sample_weight, k=2) == 1.0
    assert weighted_expected_calibration_error(probabilities, labels, sample_weight, n_bins=2) == pytest.approx(0.25)


def test_validate_sample_weight_rejects_invalid_weights() -> None:
    with pytest.raises(ValueError, match="same samples"):
        validate_sample_weight(np.array([1.0]), 2)

    with pytest.raises(ValueError, match="non-negative"):
        validate_sample_weight(np.array([1.0, -1.0]), 2)

    with pytest.raises(ValueError, match="positive total"):
        validate_sample_weight(np.array([0.0, 0.0]), 2)

    with pytest.raises(ValueError, match="finite"):
        validate_sample_weight(np.array([1.0, np.nan]), 2)


def test_weighted_probability_metrics_validate_inputs() -> None:
    probabilities = np.array([[0.7, 0.3], [0.4, 0.6]])
    labels = np.array([0, 1])
    sample_weight = np.array([1.0, 1.0])

    with pytest.raises(ValueError, match="sum to one"):
        weighted_brier_score_multiclass(np.array([[0.7, 0.7], [0.4, 0.6]]), labels, sample_weight)

    with pytest.raises(ValueError, match="valid column indices"):
        weighted_negative_log_likelihood(probabilities, np.array([0, 2]), sample_weight)

    with pytest.raises(ValueError, match="positive"):
        weighted_top_k_accuracy(probabilities, labels, sample_weight, k=0)
