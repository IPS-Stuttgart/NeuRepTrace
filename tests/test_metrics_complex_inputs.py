from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from neureptrace.metrics import (
    brier_score_multiclass,
    expected_calibration_error,
    negative_log_likelihood,
    reliability_bins,
    top_k_accuracy,
    validate_probability_inputs,
)


@pytest.mark.parametrize(
    "metric",
    [
        validate_probability_inputs,
        brier_score_multiclass,
        expected_calibration_error,
        negative_log_likelihood,
        reliability_bins,
        top_k_accuracy,
    ],
    ids=["validator", "brier", "ece", "nll", "reliability", "top-k"],
)
def test_public_probability_metrics_reject_complex_probability_arrays(metric) -> None:
    probabilities = np.array(
        [
            [0.8 + 0.1j, 0.2 - 0.1j],
            [0.3 - 0.2j, 0.7 + 0.2j],
        ],
        dtype=np.complex128,
    )

    with pytest.raises(ValueError, match="complex"):
        metric(probabilities, np.array([0, 1]))


def test_probability_validator_rejects_complex_class_indices() -> None:
    probabilities = np.array([[0.8, 0.2], [0.3, 0.7]])
    labels = np.array([0.0 + 1.0j, 1.0 + 0.0j], dtype=np.complex128)

    with pytest.raises(ValueError, match="complex"):
        validate_probability_inputs(probabilities, labels)


def test_probability_validator_rejects_complex_one_pass_rows() -> None:
    probabilities = (row for row in [[0.8 + 0.1j, 0.2 - 0.1j]])

    with pytest.raises(ValueError, match="complex"):
        validate_probability_inputs(probabilities, [0])


def _probabilities() -> np.ndarray:
    return np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=float)


def _labels() -> np.ndarray:
    return np.asarray([0, 1], dtype=int)


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (
            lambda: validate_probability_inputs(
                _probabilities(),
                normalization_atol=np.complex128(1e-6 + 1e-3j),
            ),
            "normalization_atol must be a non-negative finite value",
        ),
        (
            lambda: expected_calibration_error(
                _probabilities(),
                _labels(),
                n_bins=np.complex128(2.0 + 1.0j),
            ),
            "n_bins must be a positive integer",
        ),
        (
            lambda: top_k_accuracy(
                _probabilities(),
                _labels(),
                k=np.asarray(1.0 + 1.0j),
            ),
            "k must be a positive integer",
        ),
        (
            lambda: negative_log_likelihood(
                _probabilities(),
                _labels(),
                eps=np.complex64(1e-6 + 1e-3j),
            ),
            "eps must be a positive finite value",
        ),
    ],
)
def test_probability_metrics_reject_complex_scalar_controls(
    operation: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        operation()
