"""Materialize one-pass source-interpolation inputs before NumPy conversion."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_source_interpolation_one_pass_patch_installed"


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


def install() -> None:
    """Patch source interpolation to accept one-pass feature/label/domain inputs."""

    module = importlib.import_module("neureptrace.decoding.source_interpolation")
    if getattr(module, _PATCH_MARKER, False):
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
    setattr(module, _PATCH_MARKER, True)


__all__ = ["install"]
