"""Runtime patch for strict negative-log-likelihood epsilon validation."""

from __future__ import annotations

from collections.abc import Iterable
from functools import wraps

import numpy as np

_EPS_ERROR = "eps must be a positive finite value"


def _validate_eps(eps: object) -> float:
    if isinstance(eps, (bool, np.bool_)):
        raise ValueError(_EPS_ERROR)
    try:
        numeric = float(eps)
    except (TypeError, ValueError) as exc:
        raise ValueError(_EPS_ERROR) from exc
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
    """Install strict epsilon validation for public NLL helpers."""
    _install_onset_boolean_summary_patch()
    _install_stimulus_detection_boolean_summary_patch()

    import neureptrace.metrics as metrics
    import neureptrace.metrics.weighted as weighted_metrics

    if getattr(metrics.negative_log_likelihood, "_nll_eps_validation_patched", False):
        return

    original_negative_log_likelihood = metrics.negative_log_likelihood
    original_weighted_negative_log_likelihood = weighted_metrics.weighted_negative_log_likelihood

    @wraps(original_negative_log_likelihood)
    def negative_log_likelihood(probabilities: np.ndarray, labels: np.ndarray, *, eps: float = 1e-15) -> float:
        return original_negative_log_likelihood(probabilities, labels, eps=_validate_eps(eps))

    @wraps(original_weighted_negative_log_likelihood)
    def weighted_negative_log_likelihood(
        probabilities: np.ndarray,
        labels: np.ndarray,
        sample_weight: Iterable[float] | np.ndarray,
        *,
        eps: float = 1e-15,
    ) -> float:
        return original_weighted_negative_log_likelihood(probabilities, labels, sample_weight, eps=_validate_eps(eps))

    negative_log_likelihood._nll_eps_validation_patched = True  # type: ignore[attr-defined]
    weighted_negative_log_likelihood._nll_eps_validation_patched = True  # type: ignore[attr-defined]
    metrics.negative_log_likelihood = negative_log_likelihood
    metrics.weighted_negative_log_likelihood = weighted_negative_log_likelihood
    weighted_metrics.weighted_negative_log_likelihood = weighted_negative_log_likelihood


__all__ = ["install"]