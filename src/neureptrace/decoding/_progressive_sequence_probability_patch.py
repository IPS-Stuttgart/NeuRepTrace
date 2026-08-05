"""Harden progressive-sequence probability validation and normalization."""

from __future__ import annotations

from typing import Any

import numpy as np

from neureptrace.decoding import _progressive_sequence_core as _core


def _contains_boolean(array: np.ndarray) -> bool:
    if array.dtype.kind == "b":
        return True
    if array.dtype.kind != "O":
        return False
    return any(isinstance(value, (bool, np.bool_)) for value in array.flat)


def _contains_complex(array: np.ndarray) -> bool:
    if array.dtype.kind == "c":
        return True
    if array.dtype.kind != "O":
        return False
    return any(isinstance(value, (complex, np.complexfloating)) for value in array.flat)


def _normalize_probability_tensor(probabilities: Any) -> np.ndarray:
    """Return finite non-negative row-normalized trial probabilities.

    Validation occurs before floating-point coercion so Boolean and complex
    values cannot be silently reinterpreted. Rows are scaled by their maximum
    before summation, preventing overflow for finite values near the float64
    limit while preserving their probability ratios.
    """

    try:
        raw = np.asarray(probabilities)
    except (TypeError, ValueError) as exc:
        raise ValueError("probabilities must contain a regular numeric tensor.") from exc
    if raw.ndim != 3:
        raise ValueError("probabilities must have shape (trials, events, classes).")
    if min(raw.shape) < 1:
        raise ValueError("probabilities must be non-empty and finite.")
    if _contains_boolean(raw):
        raise ValueError("probabilities must contain real-valued numeric entries, not Boolean values.")
    if _contains_complex(raw):
        raise ValueError("probabilities must contain real-valued numeric entries, not complex values.")
    try:
        array = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("probabilities must contain real-valued numeric entries.") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError("probabilities must be non-empty and finite.")
    if np.any(array < 0.0):
        raise ValueError("probabilities must be non-negative.")

    row_maxima = np.max(array, axis=2, keepdims=True)
    if np.any(row_maxima <= 0.0):
        raise ValueError("Each event probability row must have positive mass.")
    scaled = array / row_maxima
    row_sums = scaled.sum(axis=2, keepdims=True)
    return scaled / row_sums


def install_progressive_sequence_probability_validation() -> None:
    """Install the hardened normalizer into the split implementation module."""

    _core._normalize_probability_tensor = _normalize_probability_tensor


install_progressive_sequence_probability_validation()
