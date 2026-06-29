from __future__ import annotations

import numpy as np
import pytest

from neureptrace.metrics import (
    expected_calibration_error,
    negative_log_likelihood,
    reliability_bins,
    top_k_accuracy,
    validate_probability_inputs,
)
from neureptrace.metrics.weighted import (
    weighted_expected_calibration_error,
    weighted_negative_log_likelihood,
    weighted_reliability_bins,
    weighted_top_k_accuracy,
)

PROBABILITIES = np.array([[0.7, 0.3], [0.4, 0.6]])
LABELS = np.array([0, 1])
SAMPLE_WEIGHT = np.array([1.0, 1.0])


@pytest.mark.parametrize("normalization_atol", [np.asarray(False), np.array([False])])
def test_probability_input_validation_rejects_boolean_array_tolerance(normalization_atol: object) -> None:
    with pytest.raises(ValueError, match="normalization_atol must be a non-negative finite value"):
        validate_probability_inputs(PROBABILITIES, normalization_atol=normalization_atol)


@pytest.mark.parametrize("n_bins", [np.asarray(True), np.array([True])])
def test_calibration_metrics_reject_boolean_array_bin_counts(n_bins: object) -> None:
    with pytest.raises(ValueError, match="n_bins must be a positive integer"):
        expected_calibration_error(PROBABILITIES, LABELS, n_bins=n_bins)

    with pytest.raises(ValueError, match="n_bins must be a positive integer"):
        reliability_bins(PROBABILITIES, LABELS, n_bins=n_bins)

    with pytest.raises(ValueError, match="n_bins must be a positive integer"):
        weighted_expected_calibration_error(PROBABILITIES, LABELS, SAMPLE_WEIGHT, n_bins=n_bins)

    with pytest.raises(ValueError, match="n_bins must be a positive integer"):
        weighted_reliability_bins(PROBABILITIES, LABELS, SAMPLE_WEIGHT, n_bins=n_bins)


@pytest.mark.parametrize("k", [np.asarray(True), np.array([True])])
def test_top_k_metrics_reject_boolean_array_k(k: object) -> None:
    with pytest.raises(ValueError, match="k must be a positive integer"):
        top_k_accuracy(PROBABILITIES, LABELS, k=k)

    with pytest.raises(ValueError, match="k must be a positive integer"):
        weighted_top_k_accuracy(PROBABILITIES, LABELS, SAMPLE_WEIGHT, k=k)


@pytest.mark.parametrize("eps", [np.asarray(True), np.array([True]), np.asarray(1e-6), np.array([1e-6])])
def test_nll_metrics_reject_array_eps(eps: object) -> None:
    with pytest.raises(ValueError, match="eps"):
        negative_log_likelihood(PROBABILITIES, LABELS, eps=eps)

    with pytest.raises(ValueError, match="eps"):
        weighted_negative_log_likelihood(PROBABILITIES, LABELS, SAMPLE_WEIGHT, eps=eps)
