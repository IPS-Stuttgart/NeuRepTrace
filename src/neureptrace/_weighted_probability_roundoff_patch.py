"""Align weighted probability metric validation with public unweighted metrics."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_weighted_probability_roundoff_patch_installed"
_PROBABILITY_NORMALIZATION_ATOL = 1e-6


def _validate_probability_inputs(probabilities: Any, labels: Any) -> tuple[np.ndarray, np.ndarray]:
    weighted = importlib.import_module("neureptrace.metrics.weighted")
    raw_probabilities = weighted._probability_input_array(probabilities)
    if weighted._probabilities_contain_boolean(raw_probabilities):
        raise ValueError("probabilities must contain numeric probability values, not boolean flags")
    probabilities = raw_probabilities.astype(float, copy=False)
    labels = np.asarray(labels)
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


def install() -> None:
    """Patch weighted probability metrics to tolerate tiny floating-point roundoff."""

    weighted = importlib.import_module("neureptrace.metrics.weighted")
    if getattr(weighted, _PATCH_MARKER, False):
        return
    weighted._validate_probability_inputs = _validate_probability_inputs
    setattr(weighted, _PATCH_MARKER, True)


__all__ = ["install"]
