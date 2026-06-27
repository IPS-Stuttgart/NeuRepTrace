from __future__ import annotations

import numpy as np
import pytest

from neureptrace.metrics.weighted import (
    weighted_brier_score_multiclass,
    weighted_negative_log_likelihood,
    weighted_top_k_accuracy,
)


def test_weighted_probability_metrics_accept_column_vector_labels() -> None:
    probabilities = np.array([[0.7, 0.3], [0.4, 0.6], [0.2, 0.8]])
    labels = np.array([[0], [1], [1]])
    sample_weight = np.array([1.0, 2.0, 3.0])

    assert weighted_brier_score_multiclass(probabilities, labels, sample_weight) == pytest.approx(
        np.average([0.18, 0.32, 0.08], weights=sample_weight)
    )
    assert weighted_negative_log_likelihood(probabilities, labels, sample_weight) == pytest.approx(
        -np.average(np.log([0.7, 0.6, 0.8]), weights=sample_weight)
    )
    assert weighted_top_k_accuracy(probabilities, labels, sample_weight, k=1) == pytest.approx(1.0)


def test_weighted_probability_metrics_reject_multi_column_labels() -> None:
    probabilities = np.array([[0.7, 0.3], [0.4, 0.6]])
    sample_weight = np.array([1.0, 1.0])

    with pytest.raises(ValueError, match="labels must have shape"):
        weighted_top_k_accuracy(probabilities, np.array([[0, 1], [1, 0]]), sample_weight, k=1)
