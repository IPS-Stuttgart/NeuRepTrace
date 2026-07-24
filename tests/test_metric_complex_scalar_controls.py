from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from neureptrace.metrics import (
    expected_calibration_error,
    negative_log_likelihood,
    reliability_bins,
    top_k_accuracy,
    validate_probability_inputs,
    weighted_expected_calibration_error,
    weighted_negative_log_likelihood,
    weighted_reliability_bins,
    weighted_top_k_accuracy,
)

_PROBABILITIES = np.asarray([[0.8, 0.2], [0.3, 0.7]], dtype=float)
_LABELS = np.asarray([0, 1], dtype=int)
_WEIGHTS = np.asarray([1.0, 2.0], dtype=float)


def test_validate_probability_inputs_rejects_numpy_complex_normalization_tolerance() -> None:
    with pytest.raises(ValueError, match="normalization_atol must be a non-negative finite value"):
        validate_probability_inputs(
            _PROBABILITIES,
            _LABELS,
            normalization_atol=np.complex128(1e-6 + 1j),
        )


@pytest.mark.parametrize(
    ("metric", "keyword", "value", "message"),
    [
        (
            expected_calibration_error,
            "n_bins",
            np.complex128(2 + 1j),
            "n_bins must be a positive integer",
        ),
        (
            reliability_bins,
            "n_bins",
            np.complex128(2 + 1j),
            "n_bins must be a positive integer",
        ),
        (
            negative_log_likelihood,
            "eps",
            np.complex128(1e-6 + 1j),
            "eps must be a positive finite value",
        ),
        (
            top_k_accuracy,
            "k",
            np.complex128(1 + 1j),
            "k must be a positive integer",
        ),
    ],
)
def test_unweighted_metrics_reject_numpy_complex_scalar_controls(
    metric: Callable[..., object],
    keyword: str,
    value: np.complexfloating,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        metric(_PROBABILITIES, _LABELS, **{keyword: value})


@pytest.mark.parametrize(
    ("metric", "keyword", "value", "message"),
    [
        (
            weighted_expected_calibration_error,
            "n_bins",
            np.complex128(2 + 1j),
            "n_bins must be a positive integer",
        ),
        (
            weighted_reliability_bins,
            "n_bins",
            np.complex128(2 + 1j),
            "n_bins must be a positive integer",
        ),
        (
            weighted_negative_log_likelihood,
            "eps",
            np.complex128(1e-6 + 1j),
            "eps must be a positive finite value",
        ),
        (
            weighted_top_k_accuracy,
            "k",
            np.complex128(1 + 1j),
            "k must be a positive integer",
        ),
    ],
)
def test_weighted_metrics_reject_numpy_complex_scalar_controls(
    metric: Callable[..., object],
    keyword: str,
    value: np.complexfloating,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        metric(_PROBABILITIES, _LABELS, _WEIGHTS, **{keyword: value})
