"""Patch Gaussian/Mahalanobis source-decoder numeric validation edge cases.

This module keeps strict source-only Gaussian and Mahalanobis helpers robust for
configuration scalars and feature-matrix validation.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from functools import wraps
from typing import Any

import numpy as np

_INSTALLED = False


def _numeric_scalar(value: Any, *, name: str, allow_zero: bool) -> float:
    kind = "non-negative" if allow_zero else "positive"
    message = f"{name} must be {kind} and finite."
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        return _numeric_scalar(value.item(), name=name, allow_zero=allow_zero)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(parsed) or parsed < 0.0 or (not allow_zero and parsed <= 0.0):
        raise ValueError(message)
    return parsed


def _positive_float(value: Any, *, name: str) -> float:
    return _numeric_scalar(value, name=name, allow_zero=False)


def _nonnegative_float(value: Any, *, name: str) -> float:
    return _numeric_scalar(value, name=name, allow_zero=True)


def _contains_boolean_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        if value.dtype == np.bool_:
            return True
        if value.dtype == object:
            return any(_contains_boolean_value(item) for item in value.ravel(order="C"))
        return False
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, np.generic):
        return isinstance(value.item(), (bool, np.bool_))
    if isinstance(value, Sequence):
        return any(_contains_boolean_value(item) for item in value)
    return False


def _has_boolean_feature_values(values: Any) -> bool:
    if _contains_boolean_value(values):
        return True
    try:
        array = np.asarray(values)
    except (TypeError, ValueError):
        return False
    if array.dtype == np.bool_:
        return True
    if array.dtype == object:
        return _contains_boolean_value(array)
    return False


def _reject_boolean_feature_values(values: Any, *, name: str) -> None:
    if _has_boolean_feature_values(values):
        raise ValueError(f"{name} must contain numeric feature values, not boolean flags.")


def install() -> None:
    """Install scalar config and boolean feature validation for source decoders."""

    global _INSTALLED
    if _INSTALLED:
        return

    importlib.import_module("neureptrace._source_interpolation_one_pass_patch").install()

    from neureptrace.decoding import source_gaussian, source_mahalanobis

    original_gaussian_feature_matrix = source_gaussian._feature_matrix
    original_mahalanobis_feature_matrix = source_mahalanobis._feature_matrix

    @wraps(original_gaussian_feature_matrix)
    def _gaussian_feature_matrix(values: Any, *, name: str) -> np.ndarray:
        _reject_boolean_feature_values(values, name=name)
        return original_gaussian_feature_matrix(values, name=name)

    @wraps(original_mahalanobis_feature_matrix)
    def _mahalanobis_feature_matrix(values: Any, *, name: str) -> np.ndarray:
        _reject_boolean_feature_values(values, name=name)
        return original_mahalanobis_feature_matrix(values, name=name)

    source_gaussian._positive_float = _positive_float
    source_gaussian._feature_matrix = _gaussian_feature_matrix
    source_mahalanobis._positive_float = _positive_float
    source_mahalanobis._nonnegative_float = _nonnegative_float
    source_mahalanobis._feature_matrix = _mahalanobis_feature_matrix
    _INSTALLED = True


__all__ = ["install"]
