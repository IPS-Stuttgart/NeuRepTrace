"""Runtime patch for stricter weighted-metric sample-weight validation."""

from __future__ import annotations

from collections.abc import Iterable
from functools import wraps

import numpy as np

_WEIGHT_ERROR = "sample_weight must contain numeric weights"
_MASK_SCALAR_TYPES = (type(True), np.asarray(True).dtype.type)


def _contains_mask_scalar(values: np.ndarray) -> bool:
    return any(isinstance(value, _MASK_SCALAR_TYPES) for value in values.ravel())


def _validate_no_mask_scalars(sample_weight: Iterable[float] | np.ndarray, n_samples: int) -> None:
    raw_weights = np.asarray(sample_weight, dtype=object)
    if raw_weights.ndim == 1 and raw_weights.shape[0] == n_samples and _contains_mask_scalar(raw_weights):
        raise ValueError(_WEIGHT_ERROR)


def install() -> None:
    """Install stricter validation for public weighted metric weights."""
    import neureptrace.metrics as metrics
    import neureptrace.metrics.weighted as weighted_metrics

    if getattr(weighted_metrics.validate_sample_weight, "_sample_weight_validation_patched", False):
        return

    original_validate_sample_weight = weighted_metrics.validate_sample_weight

    @wraps(original_validate_sample_weight)
    def validate_sample_weight(sample_weight: Iterable[float] | np.ndarray, n_samples: int) -> np.ndarray:
        _validate_no_mask_scalars(sample_weight, n_samples)
        return original_validate_sample_weight(sample_weight, n_samples)

    validate_sample_weight._sample_weight_validation_patched = True  # type: ignore[attr-defined]
    weighted_metrics.validate_sample_weight = validate_sample_weight
    metrics.validate_sample_weight = validate_sample_weight


__all__ = ["install"]
