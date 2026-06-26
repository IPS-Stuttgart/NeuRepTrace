"""Helpers for row-wise source-domain identifiers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _object_vector(values: Sequence[Any]) -> np.ndarray:
    vector = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        vector[index] = value
    return vector


def atomic_domain_vector(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    """Return one object-valued domain id per row without flattening composite ids."""

    array = np.asarray(values, dtype=object)
    if array.ndim == 0:
        return _object_vector([array.item()])
    if array.ndim == 1:
        return array.reshape(-1)
    if 1 in array.shape:
        return array.reshape(-1)
    rows = [tuple(row.tolist()) for row in array.reshape(array.shape[0], -1)]
    return _object_vector(rows)


def values_equal(left: object, right: object) -> bool:
    """Compare scalar or composite domain ids without ambiguous ndarray truth values."""

    try:
        equal = left == right
    except (TypeError, ValueError):
        return False
    if isinstance(equal, np.ndarray):
        return bool(np.all(equal))
    try:
        return bool(equal)
    except (TypeError, ValueError):
        return False


def ordered_unique(values: Sequence[Any] | np.ndarray) -> tuple[object, ...]:
    """Return stable first-seen unique domain ids."""

    unique: list[object] = []
    for value in atomic_domain_vector(values):
        if not any(values_equal(existing, value) for existing in unique):
            unique.append(value)
    return tuple(unique)


def domain_mask(values: Sequence[Any] | np.ndarray, selected: Sequence[object]) -> np.ndarray:
    """Return a boolean mask for rows whose domain id is in ``selected``."""

    vector = atomic_domain_vector(values)
    return np.asarray([any(values_equal(value, item) for item in selected) for value in vector], dtype=bool)


def hashable_domain_id(value: object) -> object:
    """Return a dictionary-safe representation for domain-risk summaries."""

    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list):
        return tuple(hashable_domain_id(item) for item in value)
    if isinstance(value, tuple):
        return tuple(hashable_domain_id(item) for item in value)
    return value


__all__ = [
    "atomic_domain_vector",
    "domain_mask",
    "hashable_domain_id",
    "ordered_unique",
    "values_equal",
]
