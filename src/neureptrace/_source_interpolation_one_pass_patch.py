"""Materialize one-pass source interpolation and masking inputs before NumPy conversion."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from functools import wraps
from typing import Any

import numpy as np

_INTERPOLATION_PATCH_MARKER = "_neureptrace_source_interpolation_one_pass_patch_installed"
_MASKING_PATCH_MARKER = "_neureptrace_source_masking_feature_input_patch_installed"
_SMOTE_INTERPOLATION_PATCH_MARKER = "_neureptrace_source_smote_stable_interpolation_patch_installed"


def _materialize_one_pass_iterable(value: Any) -> Any:
    """Expand generator-style containers once while preserving scalar labels."""

    if isinstance(value, np.ndarray):
        if value.dtype == object:
            if value.ndim == 0:
                return _materialize_one_pass_iterable(value.item())
            return _materialize_one_pass_iterable(value.tolist())
        return value
    if isinstance(value, (str, bytes, Mapping)):
        return value
    try:
        iterator = iter(value)
    except TypeError:
        return value
    return [_materialize_one_pass_iterable(item) for item in iterator]


def _contains_boolean_values(value: Any) -> bool:
    """Return whether a materialized feature container includes boolean values."""

    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        if value.dtype == np.bool_:
            return value.size > 0
        if value.dtype == object:
            return any(_contains_boolean_values(item) for item in value.flat)
        return False
    if isinstance(value, (str, bytes, Mapping)):
        return False
    try:
        iterator = iter(value)
    except TypeError:
        return False
    return any(_contains_boolean_values(item) for item in iterator)


def _install_source_interpolation_patch() -> None:
    module = importlib.import_module("neureptrace.decoding.source_interpolation")
    if getattr(module, _INTERPOLATION_PATCH_MARKER, False):
        return

    original_feature_matrix = module._feature_matrix
    original_value_vector = module._value_vector

    @wraps(original_feature_matrix)
    def _feature_matrix(values: Any, *, name: str) -> np.ndarray:
        return original_feature_matrix(_materialize_one_pass_iterable(values), name=name)

    @wraps(original_value_vector)
    def _value_vector(values: Any, *, expected_length: int, name: str) -> np.ndarray:
        materialized = values if isinstance(values, (str, bytes)) else _materialize_one_pass_iterable(values)
        return original_value_vector(materialized, expected_length=expected_length, name=name)

    module._feature_matrix = _feature_matrix
    module._value_vector = _value_vector
    setattr(module, _INTERPOLATION_PATCH_MARKER, True)


def _install_source_masking_patch() -> None:
    module = importlib.import_module("neureptrace.decoding.source_masking")
    if getattr(module, _MASKING_PATCH_MARKER, False):
        return

    original_feature_matrix = module._feature_matrix

    @wraps(original_feature_matrix)
    def _feature_matrix(values: Any, *, name: str) -> np.ndarray:
        materialized = _materialize_one_pass_iterable(values)
        if _contains_boolean_values(materialized):
            raise ValueError(f"{name} must contain numeric, non-boolean values.")
        return original_feature_matrix(materialized, name=name)

    module._feature_matrix = _feature_matrix
    setattr(module, _MASKING_PATCH_MARKER, True)


def _install_source_smote_interpolation_patch() -> None:
    module = importlib.import_module("neureptrace.decoding.source_smote")
    original_interpolate_rows = module.interpolate_rows
    if getattr(original_interpolate_rows, _SMOTE_INTERPOLATION_PATCH_MARKER, False):
        return

    @wraps(original_interpolate_rows)
    def interpolate_rows(content_row: Any, partner_row: Any, lam: Any) -> np.ndarray:
        left = np.asarray(content_row, dtype=float).reshape(-1)
        right = np.asarray(partner_row, dtype=float).reshape(-1)
        if left.shape != right.shape or left.size == 0:
            raise ValueError("content_row and partner_row must be non-empty vectors with the same shape.")
        weight = module._unit_interval_float(lam, name="lam")

        same_sign = np.signbit(left) == np.signbit(right)
        row = np.empty_like(left)
        row[same_sign] = left[same_sign] + weight * (right[same_sign] - left[same_sign])
        row[~same_sign] = (1.0 - weight) * left[~same_sign] + weight * right[~same_sign]
        return row.astype(np.float32, copy=False)

    setattr(interpolate_rows, _SMOTE_INTERPOLATION_PATCH_MARKER, True)
    module.interpolate_rows = interpolate_rows


def install() -> None:
    """Patch source interpolation, masking, and SMOTE numeric behavior."""

    _install_source_interpolation_patch()
    _install_source_masking_patch()
    _install_source_smote_interpolation_patch()


__all__ = ["install"]
