"""Align weighted probability metric validation with public unweighted metrics."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_weighted_probability_roundoff_patch_installed"
_PROBABILITY_NORMALIZATION_ATOL = 1e-6
_WEIGHTED_REDUCTION_SAFETY_FACTOR = 1024.0
_COMPLEX_PROBABILITY_ERROR = "probabilities must contain real-valued probability values, not complex values"
_COMPLEX_LABEL_ERROR = "labels must contain real integer class indices, not complex values"


def _label_input_array(labels: Any) -> np.ndarray:
    """Return labels as an array without exhausting one-pass iterables implicitly."""

    if isinstance(labels, np.ndarray) or isinstance(labels, (str, bytes)):
        return np.asarray(labels)
    try:
        return np.asarray(list(labels))
    except TypeError:
        return np.asarray(labels)
    except ValueError as exc:
        raise ValueError("labels must have shape (n_samples,)") from exc


def _contains_complex(value: object) -> bool:
    """Return whether a materialized weighted-metric input is complex-valued."""

    if isinstance(value, (complex, np.complexfloating)):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.complexfloating):
            return bool(value.size)
        if value.dtype == object:
            return any(_contains_complex(item) for item in value.ravel(order="C"))
    return False


def _validate_probability_inputs(probabilities: Any, labels: Any) -> tuple[np.ndarray, np.ndarray]:
    weighted = importlib.import_module("neureptrace.metrics.weighted")
    raw_probabilities = weighted._probability_input_array(probabilities)
    raw_labels = _label_input_array(labels)
    if _contains_complex(raw_probabilities):
        raise ValueError(_COMPLEX_PROBABILITY_ERROR)
    if _contains_complex(raw_labels):
        raise ValueError(_COMPLEX_LABEL_ERROR)
    if weighted._probabilities_contain_boolean(raw_probabilities):
        raise ValueError("probabilities must contain numeric probability values, not boolean flags")
    probabilities = raw_probabilities.astype(float, copy=False)
    labels = raw_labels
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (n_samples, n_classes)")
    if probabilities.shape[0] == 0 or probabilities.shape[1] == 0:
        raise ValueError("probabilities must contain at least one sample and one class")
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels.reshape(-1)
    if labels.ndim != 1:
        raise ValueError("labels must have shape (n_samples,)")
    if probabilities.shape[0] != labels.shape[0]:
        raise ValueError("probabilities and labels must contain the same samples")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must contain only finite values")
    if np.any(probabilities < -_PROBABILITY_NORMALIZATION_ATOL):
        raise ValueError("probabilities must be non-negative")
    if np.any(probabilities < 0.0):
        probabilities = np.maximum(probabilities, 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        row_sums = probabilities.sum(axis=1)
    if not np.all(np.isfinite(row_sums)) or not np.allclose(row_sums, 1.0, atol=_PROBABILITY_NORMALIZATION_ATOL, rtol=0.0):
        raise ValueError("probability rows must sum to one")
    probabilities = probabilities / row_sums[:, None]
    labels = weighted._coerce_label_indices(labels)
    if np.any(labels < 0) or np.any(labels >= probabilities.shape[1]):
        raise ValueError("labels must be valid column indices for probabilities")
    return probabilities, labels


def _overflow_safe_sample_weight_validator(original_validate):
    """Wrap sample-weight validation with scale-invariant reduction protection."""

    @wraps(original_validate)
    def validate_sample_weight(sample_weight: Any, n_samples: int) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            weights = original_validate(sample_weight, n_samples)
            total_weight = float(np.sum(weights))
        float_info = np.finfo(float)
        safe_max_total = float_info.max / _WEIGHTED_REDUCTION_SAFETY_FACTOR
        safe_min_total = float_info.tiny * _WEIGHTED_REDUCTION_SAFETY_FACTOR
        if np.isfinite(total_weight) and safe_min_total <= total_weight <= safe_max_total:
            return weights

        max_weight = float(np.max(weights))
        # The original validator guarantees non-negative weights and positive
        # total mass, so max_weight is finite and strictly positive here.
        # Rescaling protects both overflow for very large weights and product
        # underflow for subnormal weights while preserving all relative weights.
        return weights / max_weight

    return validate_sample_weight


def install() -> None:
    """Patch weighted metrics for probability roundoff and stable weight reductions."""

    weighted = importlib.import_module("neureptrace.metrics.weighted")
    if getattr(weighted, _PATCH_MARKER, False):
        return

    weighted._validate_probability_inputs = _validate_probability_inputs
    validate_sample_weight = _overflow_safe_sample_weight_validator(weighted.validate_sample_weight)
    weighted.validate_sample_weight = validate_sample_weight

    # ``neureptrace.metrics`` binds the public helper before installing this
    # compatibility patch, so keep that export aligned with the patched module.
    metrics = importlib.import_module("neureptrace.metrics")
    metrics.validate_sample_weight = validate_sample_weight
    setattr(weighted, _PATCH_MARKER, True)


__all__ = ["install"]
