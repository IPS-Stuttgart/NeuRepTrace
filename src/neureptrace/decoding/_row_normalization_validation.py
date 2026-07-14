"""Shared input validation for stateless row normalizers."""

from __future__ import annotations

from typing import Any

import numpy as np


def feature_matrix(values: Any, *, name: str) -> np.ndarray:
    """Return a finite non-boolean two-dimensional numeric feature matrix."""

    materialized = _materialize_iterables(values)
    if _contains_boolean_value(materialized):
        raise ValueError(f"{name} must not contain boolean values.")
    try:
        matrix = np.asarray(materialized, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a non-empty two-dimensional numeric matrix."
        ) from exc
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def positive_float(value: float | str, *, name: str) -> float:
    """Return a positive finite scalar without accepting boolean values."""

    if _is_boolean_scalar(value):
        raise ValueError(f"{name} must be positive and finite.")
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"{name} must be positive and finite.")
        value = value.item()
        if _is_boolean_scalar(value):
            raise ValueError(f"{name} must be positive and finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite.") from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed


def _materialize_iterables(values: Any) -> Any:
    if isinstance(values, np.ndarray) or isinstance(values, (str, bytes)):
        return values
    try:
        iterator = iter(values)
    except TypeError:
        return values
    return [_materialize_iterables(item) for item in iterator]


def _contains_boolean_value(values: Any) -> bool:
    if isinstance(values, (bool, np.bool_)):
        return True
    if isinstance(values, np.ndarray):
        if np.issubdtype(values.dtype, np.bool_):
            return True
        if values.dtype == object:
            return any(_contains_boolean_value(item) for item in values.flat)
        return False
    if isinstance(values, (str, bytes)):
        return False
    try:
        iterator = iter(values)
    except TypeError:
        return False
    return any(_contains_boolean_value(item) for item in iterator)


def _is_boolean_scalar(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray) and value.shape == ():
        return isinstance(value.item(), (bool, np.bool_))
    return False
