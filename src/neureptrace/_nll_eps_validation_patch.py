"""Runtime patch for strict public metric scalar validation."""

from __future__ import annotations

from collections.abc import Iterable
from functools import wraps

import numpy as np

_EPS_ERROR = "eps must be a positive finite value"


def _coerce_non_array_scalar(value: object, message: str) -> float:
    if isinstance(value, np.ndarray):
        raise ValueError(message)
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc


def _validate_non_negative_finite_float(value: object, name: str) -> float:
    message = f"{name} must be a non-negative finite value"
    numeric = _coerce_non_array_scalar(value, message)
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(message)
    return numeric


def _validate_positive_integer(value: object, name: str) -> int:
    message = f"{name} must be a positive integer"
    numeric = _coerce_non_array_scalar(value, message)
    if not np.isfinite(numeric) or numeric < 1.0 or numeric % 1.0 != 0.0:
        raise ValueError(message)
    return int(numeric)


def _validate_eps(eps: object) -> float:
    numeric = _coerce_non_array_scalar(eps, _EPS_ERROR)
    if not np.isfinite(numeric) or numeric <= 0.0 or numeric >= 1.0:
        raise ValueError(_EPS_ERROR)
    return numeric


def _install_onset_boolean_summary_patch() -> None:
    from neureptrace import _onset_boolean_summary_patch

    _onset_boolean_summary_patch.install()


def _install_stimulus_detection_boolean_summary_patch() -> None:
    from neureptrace import _stimulus_detection_boolean_summary_patch

    _stimulus_detection_boolean_summary_patch.install()


def install() -> None:
    """Install strict scalar validation for public probability metrics."""
    _install_onset_boolean_summary_patch()
    _install_stimulus_detection_boolean_summary_patch()

    import neureptrace.metrics as metrics
    import neureptrace.metrics.weighted as weighted_metrics

    if getattr(metrics.validate_probability_inputs, "_metric_scalar_array_validation_patched", False):
        return

    original_validate_probability_inputs = metrics.validate_probability_inputs
    original_expected_calibration_error = metrics.expected_calibration_error
    original_reliability_bins = metrics.reliability_bins
    original_negative_log_likelihood = metrics.negative_log_likelihood
    original_top_k_accuracy = metrics.top_k_accuracy
    original_weighted_expected_calibration_error = weighted_metrics.weighted_expected_calibration_error
    original_weighted_reliability_bins = weighted_metrics.weighted_reliability_bins
    original_weighted_negative_log_likelihood = weighted_metrics.weighted_negative_log_likelihood
    original_weighted_top_k_accuracy = weighted_metrics.weighted_top_k_accuracy

    @wraps(original_validate_probability_inputs)
    def validate_probability_inputs(
        probabilities: np.ndarray,
        labels: np.ndarray | None = None,
        *,
        require_normalized: bool = True,
        normalization_atol: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        return original_validate_probability_inputs(
            probabilities,
            labels,
            require_normalized=require_normalized,
            normalization_atol=_validate_non_negative_finite_float(normalization_atol, "normalization_atol"),
        )

    @wraps(original_expected_calibration_error)
    def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, *, n_bins: int = 10) -> float:
        return original_expected_calibration_error(
            probabilities,
            labels,
            n_bins=_validate_positive_integer(n_bins, "n_bins"),
        )

    @wraps(original_reliability_bins)
    def reliability_bins(probabilities: np.ndarray, labels: np.ndarray, *, n_bins: int = 10) -> list[dict[str, float | int]]:
        return original_reliability_bins(
            probabilities,
            labels,
            n_bins=_validate_positive_integer(n_bins, "n_bins"),
        )

    @wraps(original_negative_log_likelihood)
    def negative_log_likelihood(probabilities: np.ndarray, labels: np.ndarray, *, eps: float = 1e-15) -> float:
        return original_negative_log_likelihood(probabilities, labels, eps=_validate_eps(eps))

    @wraps(original_top_k_accuracy)
    def top_k_accuracy(probabilities: np.ndarray, labels: np.ndarray, *, k: int = 1) -> float:
        return original_top_k_accuracy(probabilities, labels, k=_validate_positive_integer(k, "k"))

    @wraps(original_weighted_expected_calibration_error)
    def weighted_expected_calibration_error(
        probabilities: np.ndarray,
        labels: np.ndarray,
        sample_weight: Iterable[float] | np.ndarray,
        *,
        n_bins: int = 10,
    ) -> float:
        return original_weighted_expected_calibration_error(
            probabilities,
            labels,
            sample_weight,
            n_bins=_validate_positive_integer(n_bins, "n_bins"),
        )

    @wraps(original_weighted_reliability_bins)
    def weighted_reliability_bins(
        probabilities: np.ndarray,
        labels: np.ndarray,
        sample_weight: Iterable[float] | np.ndarray,
        *,
        n_bins: int = 10,
    ) -> list[dict[str, float | int]]:
        return original_weighted_reliability_bins(
            probabilities,
            labels,
            sample_weight,
            n_bins=_validate_positive_integer(n_bins, "n_bins"),
        )

    @wraps(original_weighted_negative_log_likelihood)
    def weighted_negative_log_likelihood(
        probabilities: np.ndarray,
        labels: np.ndarray,
        sample_weight: Iterable[float] | np.ndarray,
        *,
        eps: float = 1e-15,
    ) -> float:
        return original_weighted_negative_log_likelihood(probabilities, labels, sample_weight, eps=_validate_eps(eps))

    @wraps(original_weighted_top_k_accuracy)
    def weighted_top_k_accuracy(
        probabilities: np.ndarray,
        labels: np.ndarray,
        sample_weight: Iterable[float] | np.ndarray,
        *,
        k: int = 1,
    ) -> float:
        return original_weighted_top_k_accuracy(
            probabilities,
            labels,
            sample_weight,
            k=_validate_positive_integer(k, "k"),
        )

    for patched in (
        validate_probability_inputs,
        expected_calibration_error,
        reliability_bins,
        negative_log_likelihood,
        top_k_accuracy,
        weighted_expected_calibration_error,
        weighted_reliability_bins,
        weighted_negative_log_likelihood,
        weighted_top_k_accuracy,
    ):
        patched._metric_scalar_array_validation_patched = True  # type: ignore[attr-defined]

    negative_log_likelihood._nll_eps_validation_patched = True  # type: ignore[attr-defined]
    weighted_negative_log_likelihood._nll_eps_validation_patched = True  # type: ignore[attr-defined]

    metrics.validate_probability_inputs = validate_probability_inputs
    metrics.expected_calibration_error = expected_calibration_error
    metrics.reliability_bins = reliability_bins
    metrics.negative_log_likelihood = negative_log_likelihood
    metrics.top_k_accuracy = top_k_accuracy
    metrics.weighted_expected_calibration_error = weighted_expected_calibration_error
    metrics.weighted_reliability_bins = weighted_reliability_bins
    metrics.weighted_negative_log_likelihood = weighted_negative_log_likelihood
    metrics.weighted_top_k_accuracy = weighted_top_k_accuracy
    weighted_metrics.weighted_expected_calibration_error = weighted_expected_calibration_error
    weighted_metrics.weighted_reliability_bins = weighted_reliability_bins
    weighted_metrics.weighted_negative_log_likelihood = weighted_negative_log_likelihood
    weighted_metrics.weighted_top_k_accuracy = weighted_top_k_accuracy


__all__ = ["install"]
