from __future__ import annotations

import numpy as np
import pytest

from neureptrace.metrics import (
    validate_sample_weight,
    weighted_brier_score_multiclass,
    weighted_expected_calibration_error,
    weighted_negative_log_likelihood,
    weighted_reliability_bins,
    weighted_top_k_accuracy,
)


@pytest.mark.parametrize(
    "metric",
    [
        weighted_brier_score_multiclass,
        weighted_expected_calibration_error,
        weighted_negative_log_likelihood,
        weighted_reliability_bins,
        weighted_top_k_accuracy,
    ],
    ids=["brier", "ece", "nll", "reliability", "top-k"],
)
def test_weighted_probability_metrics_reject_complex_sample_weights(metric) -> None:
    probabilities = np.asarray([[0.8, 0.2], [0.3, 0.7]])
    labels = np.asarray([0, 1])
    sample_weight = np.asarray(
        [np.complex128(1.0 + 2.0j), np.complex64(2.0 + 3.0j)],
        dtype=object,
    )

    with pytest.raises(ValueError, match="sample_weight must contain real-valued numeric weights"):
        metric(probabilities, labels, sample_weight)


def test_validate_sample_weight_rejects_zero_imaginary_complex_scalars() -> None:
    sample_weight = np.asarray(
        [np.complex128(1.0 + 0.0j), np.complex64(2.0 + 0.0j)],
        dtype=object,
    )

    with pytest.raises(ValueError, match="sample_weight must contain real-valued numeric weights"):
        validate_sample_weight(sample_weight, 2)


def test_weighted_top_k_rejects_complex_k_before_float_narrowing() -> None:
    probabilities = np.asarray([[0.8, 0.2], [0.3, 0.7]])
    labels = np.asarray([0, 1])
    sample_weight = np.ones(2)

    with pytest.raises(ValueError, match="k must be a positive integer"):
        weighted_top_k_accuracy(
            probabilities,
            labels,
            sample_weight,
            k=np.complex128(1.0 + 4.0j),
        )


@pytest.mark.parametrize(
    "metric",
    [weighted_expected_calibration_error, weighted_reliability_bins],
    ids=["ece", "reliability"],
)
def test_weighted_calibration_metrics_reject_complex_n_bins(metric) -> None:
    probabilities = np.asarray([[0.8, 0.2], [0.3, 0.7]])
    labels = np.asarray([0, 1])
    sample_weight = np.ones(2)

    with pytest.raises(ValueError, match="n_bins must be a positive integer"):
        metric(
            probabilities,
            labels,
            sample_weight,
            n_bins=np.complex64(2.0 + 3.0j),
        )


def test_weighted_nll_rejects_complex_eps_before_float_narrowing() -> None:
    probabilities = np.asarray([[0.8, 0.2], [0.3, 0.7]])
    labels = np.asarray([0, 1])
    sample_weight = np.ones(2)

    with pytest.raises(ValueError, match="eps must be finite"):
        weighted_negative_log_likelihood(
            probabilities,
            labels,
            sample_weight,
            eps=np.complex128(1e-3 + 0.5j),
        )
