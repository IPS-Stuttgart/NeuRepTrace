"""Scale-stable normalization for source and target prior adjustment."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_prior_probability_normalization_overflow_patch_installed"
_SOURCE_FINITE_ERROR = "probability rows must be finite and non-negative."
_SOURCE_MASS_ERROR = "probability rows must have positive mass."
_TARGET_FINITE_ERROR = "probability rows must be finite and non-negative."
_TARGET_MASS_ERROR = "probability rows must have positive mass."
_PRIOR_ERROR = "source_prior must contain finite non-negative values with positive mass."


def _stable_probability_rows(
    values: Any,
    *,
    epsilon: float,
    clip_to_epsilon: bool,
    finite_error: str,
    mass_error: str,
) -> np.ndarray:
    """Normalize finite non-negative rows without summing raw magnitudes."""

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError(finite_error)
    if clip_to_epsilon:
        matrix = np.maximum(matrix, float(epsilon))
    if matrix.shape[1] < 1:
        raise ValueError(mass_error)

    row_maxima = np.max(matrix, axis=1, keepdims=True)
    if np.any(row_maxima <= 0.0):
        raise ValueError(mass_error)
    scaled = matrix / row_maxima
    scaled_sums = np.sum(scaled, axis=1, keepdims=True)
    return scaled / scaled_sums


def _stable_probability_vector(values: Any, *, epsilon: float) -> np.ndarray:
    """Normalize a prior vector without overflowing its raw sum."""

    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.shape[0] < 1 or not np.all(np.isfinite(vector)) or np.any(vector < 0.0):
        raise ValueError("prior values must be finite and non-negative.")
    vector = np.maximum(vector, float(epsilon))
    maximum = float(np.max(vector))
    scaled = vector / maximum
    return scaled / float(np.sum(scaled))


def install() -> None:
    """Install scale-stable probability normalization in prior adjusters."""

    source_prior = importlib.import_module("neureptrace.decoding.source_prior")
    if not getattr(source_prior._normalize_probability_rows, _PATCH_MARKER, False):
        original_source_rows = source_prior._normalize_probability_rows

        @wraps(original_source_rows)
        def _normalize_source_probability_rows(values: np.ndarray, *, epsilon: float) -> np.ndarray:
            return _stable_probability_rows(
                values,
                epsilon=epsilon,
                clip_to_epsilon=False,
                finite_error=_SOURCE_FINITE_ERROR,
                mass_error=_SOURCE_MASS_ERROR,
            )

        setattr(_normalize_source_probability_rows, _PATCH_MARKER, True)
        source_prior._normalize_probability_rows = _normalize_source_probability_rows

    if not getattr(source_prior._normalize_probability_vector, _PATCH_MARKER, False):
        original_source_vector = source_prior._normalize_probability_vector

        @wraps(original_source_vector)
        def _normalize_source_probability_vector(values: np.ndarray, *, epsilon: float) -> np.ndarray:
            return _stable_probability_vector(values, epsilon=epsilon)

        setattr(_normalize_source_probability_vector, _PATCH_MARKER, True)
        source_prior._normalize_probability_vector = _normalize_source_probability_vector

    target_prior = importlib.import_module("neureptrace.decoding.target_prior_adjustment")
    if not getattr(target_prior._normalize_probability_rows, _PATCH_MARKER, False):
        original_target_rows = target_prior._normalize_probability_rows

        @wraps(original_target_rows)
        def _normalize_target_probability_rows(values: np.ndarray, *, epsilon: float) -> np.ndarray:
            return _stable_probability_rows(
                values,
                epsilon=epsilon,
                clip_to_epsilon=True,
                finite_error=_TARGET_FINITE_ERROR,
                mass_error=_TARGET_MASS_ERROR,
            )

        setattr(_normalize_target_probability_rows, _PATCH_MARKER, True)
        target_prior._normalize_probability_rows = _normalize_target_probability_rows

    if not getattr(target_prior._prior_vector, _PATCH_MARKER, False):
        original_prior_vector = target_prior._prior_vector

        @wraps(original_prior_vector)
        def _prior_vector(value: Any, *, n_classes: int, epsilon: float) -> np.ndarray:
            prior = np.asarray(value, dtype=float).reshape(-1)
            if prior.shape[0] != n_classes:
                raise ValueError(f"source_prior must contain one value per class: {prior.shape[0]} != {n_classes}.")
            if not np.all(np.isfinite(prior)) or np.any(prior < 0.0) or not bool(np.any(prior > 0.0)):
                raise ValueError(_PRIOR_ERROR)
            return target_prior._normalize_probability_rows(prior[None, :], epsilon=epsilon)[0]

        setattr(_prior_vector, _PATCH_MARKER, True)
        target_prior._prior_vector = _prior_vector


__all__ = ["install"]
