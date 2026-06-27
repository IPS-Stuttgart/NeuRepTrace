"""Preserve composite labels/domains in source-domain generalization encoders."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

_PATCH_MARKER = "_neureptrace_source_domain_generalization_composite_patch_installed"


def _object_value_vector(values: Sequence[Any]) -> np.ndarray:
    vector = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        vector[index] = value
    return vector


def _atomic_vector(values: Sequence[Any] | np.ndarray, *, name: str, reject_matrix: bool = False) -> np.ndarray:
    """Normalize row labels while keeping tuple/list IDs as one value per row."""

    array = np.asarray(values, dtype=object)
    if array.ndim == 0:
        return _object_value_vector([array.item()])
    if array.ndim == 1:
        return array.reshape(-1)
    if reject_matrix and min(array.shape) > 1:
        raise ValueError(f"{name} must be a one-dimensional vector, not a matrix-shaped array.")
    if 1 in array.shape:
        return array.reshape(-1)
    rows = [tuple(row.tolist()) for row in array.reshape(array.shape[0], -1)]
    return _object_value_vector(rows)


def _values_equal(left: object, right: object) -> bool:
    try:
        equal = left == right
    except (TypeError, ValueError):
        return False
    try:
        return bool(equal)
    except (TypeError, ValueError):
        return False


def _is_missing_domain_value(value: Any) -> bool:
    """Return true when a scalar or composite source-domain id is missing."""

    if value is None:
        return True
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _is_missing_domain_value(value.item())
        return any(_is_missing_domain_value(item) for item in value.reshape(-1).tolist())
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_is_missing_domain_value(item) for item in value)
    if isinstance(value, dict):
        return any(_is_missing_domain_value(key) or _is_missing_domain_value(item) for key, item in value.items())

    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    try:
        return bool(np.any(missing))
    except (TypeError, ValueError):
        return False


def _is_missing_domain_array(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    """Vectorized missing-domain detector that also handles composite IDs."""

    flattened = np.asarray(values, dtype=object).reshape(-1)
    return np.asarray([_is_missing_domain_value(value) for value in flattened], dtype=bool)


def _ordered_unique(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    unique: list[object] = []
    for value in _atomic_vector(values, name="values"):
        if not any(_values_equal(existing, value) for existing in unique):
            unique.append(value)
    return _object_value_vector(unique)


def _encode_atomic(values: Sequence[Any] | np.ndarray, *, name: str, reject_matrix: bool = False) -> tuple[np.ndarray, np.ndarray]:
    vector = _atomic_vector(values, name=name, reject_matrix=reject_matrix)
    unique = _ordered_unique(vector)
    encoded = np.zeros(vector.shape[0], dtype=np.int64)
    for code, value in enumerate(unique):
        encoded[np.asarray([_values_equal(item, value) for item in vector], dtype=bool)] = code
    return unique, encoded


def install() -> None:
    """Patch source-domain generalization input encoding."""

    module = importlib.import_module("neureptrace.decoding.source_domain_generalization")
    if getattr(module, _PATCH_MARKER, False):
        return

    module._is_missing_domain_array = _is_missing_domain_array

    def _encode_inputs(source_features: np.ndarray, source_labels: np.ndarray, source_domains: np.ndarray, *, name: str):
        x = np.asarray(source_features, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"{name} source_features must be two-dimensional.")
        if x.shape[0] < 2:
            raise ValueError(f"{name} needs at least two source rows.")

        classes, y = _encode_atomic(source_labels, name="source_labels")
        if y.shape[0] != x.shape[0]:
            raise ValueError("source_features and source_labels must contain the same rows.")
        if classes.shape[0] < 2:
            raise ValueError(f"{name} needs at least two source classes.")

        domain_names, domains = _encode_atomic(source_domains, name="source_domains")
        if domains.shape[0] != x.shape[0]:
            raise ValueError("source_features and source_domains must contain the same rows.")
        if np.any(module._is_missing_domain_array(domain_names)):
            raise ValueError("source_domains must not contain missing values.")
        if domain_names.shape[0] < 2:
            raise ValueError(f"{name} needs at least two source domains/subjects.")
        return x, classes, y.astype(np.int64, copy=False), domain_names, domains.astype(np.int64, copy=False)

    module._encode_inputs = _encode_inputs
    setattr(module, _PATCH_MARKER, True)


__all__ = ["install"]
