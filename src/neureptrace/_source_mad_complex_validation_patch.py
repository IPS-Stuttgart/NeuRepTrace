"""Reject complex feature inputs in source-only MAD normalization."""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_source_mad_complex_validation_patch_installed"


def _contains_complex(value: object) -> bool:
    """Return whether a materialized feature input contains complex values."""

    if isinstance(value, (complex, np.complexfloating)):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.complexfloating):
            return bool(value.size)
        if value.dtype == object:
            return any(_contains_complex(item) for item in value.ravel(order="C"))
        return False
    if isinstance(value, (str, bytes)):
        return False
    if hasattr(value, "__array__"):
        try:
            return _contains_complex(np.asarray(value))
        except (TypeError, ValueError):
            return False
    if not isinstance(value, Iterable):
        return False
    return any(_contains_complex(item) for item in value)


def install() -> None:
    """Install real-valued feature validation for source-MAD APIs."""

    source_mad = importlib.import_module("neureptrace.decoding.source_mad")
    original_matrix = source_mad._matrix
    if getattr(original_matrix, _PATCH_MARKER, False):
        return

    @wraps(original_matrix)
    def _matrix(values: Any, *, name: str) -> np.ndarray:
        materialized = source_mad._materialize_feature_iterables(values)
        if _contains_complex(materialized):
            raise ValueError(f"{name} must contain real-valued feature values, not complex values.")
        return original_matrix(materialized, name=name)

    setattr(_matrix, _PATCH_MARKER, True)
    _matrix.__wrapped__ = original_matrix
    source_mad._matrix = _matrix


__all__ = ["install"]
