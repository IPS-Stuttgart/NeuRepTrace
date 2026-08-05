"""Harden Source Feature Roll numeric feature inputs."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np

_MATRIX_MARKER = "_source_roll_numeric_matrix_inputs_patched"
_ROW_MARKER = "_source_roll_numeric_row_inputs_patched"


def _materialize_nested(value: Any) -> Any:
    """Materialize one-pass iterators without changing reusable containers."""

    if isinstance(value, np.ndarray):
        if value.dtype != object:
            return value
        materialized = np.empty(value.shape, dtype=object)
        for index in np.ndindex(value.shape):
            materialized[index] = _materialize_nested(value[index])
        return materialized
    if isinstance(value, Iterator):
        return tuple(_materialize_nested(item) for item in value)
    if isinstance(value, list):
        return [_materialize_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_materialize_nested(item) for item in value)
    return value


def _contains_boolean(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.bool_):
            return True
        if value.dtype == object:
            return any(_contains_boolean(item) for item in value.ravel(order="C"))
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_boolean(item) for item in value)
    return False


def _contains_complex(value: Any) -> bool:
    if isinstance(value, (complex, np.complexfloating)):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.complexfloating):
            return True
        if value.dtype == object:
            return any(_contains_complex(item) for item in value.ravel(order="C"))
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_complex(item) for item in value)
    return False


def _validate_numeric_features(value: Any, *, name: str) -> Any:
    materialized = _materialize_nested(value)
    if _contains_boolean(materialized):
        raise ValueError(f"{name} must contain real numeric values, not booleans.")
    if _contains_complex(materialized):
        raise ValueError(f"{name} must contain real-valued features, not complex values.")
    return materialized


def _installed_here(function: Any, marker: str) -> bool:
    """Distinguish this guard from outer wrappers that copied its marker."""

    attributes = getattr(function, "__dict__", {})
    code = getattr(function, "__code__", None)
    if not attributes.get(marker, False) or code is None:
        return False
    return Path(code.co_filename).resolve() == Path(__file__).resolve()


def install() -> None:
    """Install Source Feature Roll feature-input guards."""

    source_roll = importlib.import_module("neureptrace.decoding.source_roll")

    original_feature_matrix = source_roll._feature_matrix
    if not _installed_here(original_feature_matrix, _MATRIX_MARKER):

        @wraps(original_feature_matrix)
        def feature_matrix(values: Any, *, name: str) -> np.ndarray:
            materialized = _validate_numeric_features(values, name=name)
            try:
                return original_feature_matrix(materialized, name=name)
            except (TypeError, OverflowError) as exc:
                raise ValueError(f"{name} must be a numeric feature matrix.") from exc

        setattr(feature_matrix, _MATRIX_MARKER, True)
        source_roll._feature_matrix = feature_matrix

    original_roll_feature_row = source_roll.roll_feature_row
    if not _installed_here(original_roll_feature_row, _ROW_MARKER):

        @wraps(original_roll_feature_row)
        def roll_feature_row(
            row: Any,
            *,
            shift: Any,
            mode: str = "circular",
            fill_value: Any = 0.0,
        ) -> np.ndarray:
            materialized = _validate_numeric_features(row, name="row")
            try:
                return original_roll_feature_row(
                    materialized,
                    shift=shift,
                    mode=mode,
                    fill_value=fill_value,
                )
            except (TypeError, OverflowError) as exc:
                raise ValueError("row must contain numeric feature values.") from exc

        setattr(roll_feature_row, _ROW_MARKER, True)
        source_roll.roll_feature_row = roll_feature_row


__all__ = ["install"]
