"""Align weighted probability metric validation with public unweighted metrics."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_weighted_probability_roundoff_patch_installed"
_PROBABILITY_NORMALIZATION_ATOL = 1e-6


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


def _validate_probability_inputs(probabilities: Any, labels: Any) -> tuple[np.ndarray, np.ndarray]:
    weighted = importlib.import_module("neureptrace.metrics.weighted")
    raw_probabilities = weighted._probability_input_array(probabilities)
    if weighted._probabilities_contain_boolean(raw_probabilities):
        raise ValueError("probabilities must contain numeric probability values, not boolean flags")
    probabilities = raw_probabilities.astype(float, copy=False)
    labels = _label_input_array(labels)
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
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=_PROBABILITY_NORMALIZATION_ATOL, rtol=0.0):
        raise ValueError("probability rows must sum to one")
    probabilities = probabilities / row_sums[:, None]
    labels = weighted._coerce_label_indices(labels)
    if np.any(labels < 0) or np.any(labels >= probabilities.shape[1]):
        raise ValueError("labels must be valid column indices for probabilities")
    return probabilities, labels


def _overflow_safe_sample_weight_validator(original_validate):
    """Wrap sample-weight validation with scale-invariant overflow protection."""

    @wraps(original_validate)
    def validate_sample_weight(sample_weight: Any, n_samples: int) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore"):
            weights = original_validate(sample_weight, n_samples)
            total_weight = float(np.sum(weights))
        if np.isfinite(total_weight):
            return weights

        max_weight = float(np.max(weights))
        # The original validator guarantees non-negative weights and positive
        # total mass, so max_weight is finite and strictly positive here.
        return weights / max_weight

    return validate_sample_weight


def install() -> None:
    """Patch weighted metrics for probability roundoff and weight-sum overflow."""

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
