"""Runtime patch for stricter weighted-metric sample-weight validation."""

from __future__ import annotations

from collections.abc import Iterable
from functools import wraps

import numpy as np

_WEIGHT_ERROR = "sample_weight must contain numeric weights, not boolean values"
_COMPLEX_WEIGHT_ERROR = "sample_weight must contain real-valued weights, not complex values"
_MASK_SCALAR_TYPES = (bool, np.bool_)
_COMPLEX_SCALAR_TYPES = (complex, np.complexfloating)


def _is_mask_scalar(value: object) -> bool:
    if isinstance(value, _MASK_SCALAR_TYPES):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.bool_):
            return True
        if value.dtype == object:
            return any(_is_mask_scalar(item) for item in value.ravel())
    return False


def _is_complex_scalar(value: object) -> bool:
    if isinstance(value, _COMPLEX_SCALAR_TYPES):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.complexfloating):
            return bool(value.size)
        if value.dtype == object:
            return any(_is_complex_scalar(item) for item in value.ravel())
    return False


def _contains_mask_scalar(values: np.ndarray) -> bool:
    return any(_is_mask_scalar(value) for value in values.ravel())


def _contains_complex_scalar(values: np.ndarray) -> bool:
    return any(_is_complex_scalar(value) for value in values.ravel())


def _looks_like_valid_weight_shape(raw_weights: np.ndarray, n_samples: int) -> bool:
    if raw_weights.ndim == 1:
        return raw_weights.shape[0] == n_samples
    if raw_weights.ndim == 2:
        return raw_weights.shape == (n_samples, 1)
    return False


def _validate_no_mask_scalars(sample_weight: Iterable[float] | np.ndarray, n_samples: int) -> None:
    raw_weights = np.asarray(sample_weight, dtype=object)
    if _looks_like_valid_weight_shape(raw_weights, n_samples) and _contains_mask_scalar(raw_weights):
        raise ValueError(_WEIGHT_ERROR)


def _validate_no_complex_scalars(sample_weight: Iterable[float] | np.ndarray, n_samples: int) -> None:
    raw_weights = np.asarray(sample_weight, dtype=object)
    if _looks_like_valid_weight_shape(raw_weights, n_samples) and _contains_complex_scalar(raw_weights):
        raise ValueError(_COMPLEX_WEIGHT_ERROR)


def install() -> None:
    """Install stricter validation for public weighted metric weights."""
    import neureptrace.metrics as metrics
    import neureptrace.metrics.weighted as weighted_metrics

    if getattr(weighted_metrics.validate_sample_weight, "_sample_weight_validation_patched", False):
        return

    original_validate_sample_weight = weighted_metrics.validate_sample_weight

    @wraps(original_validate_sample_weight)
    def validate_sample_weight(sample_weight: Iterable[float] | np.ndarray, n_samples: int) -> np.ndarray:
        raw_weights = weighted_metrics._sample_weight_array(sample_weight)
        _validate_no_mask_scalars(raw_weights, n_samples)
        _validate_no_complex_scalars(raw_weights, n_samples)
        return original_validate_sample_weight(raw_weights, n_samples)

    validate_sample_weight._sample_weight_validation_patched = True  # type: ignore[attr-defined]
    weighted_metrics.validate_sample_weight = validate_sample_weight
    metrics.validate_sample_weight = validate_sample_weight


__all__ = ["install"]
