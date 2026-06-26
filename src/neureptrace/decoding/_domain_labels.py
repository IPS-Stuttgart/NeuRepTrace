"""Helpers for preserving scalar and composite source-domain labels."""

from __future__ import annotations

from collections.abc import Mapping, Set
from typing import Any

import numpy as np


def _as_domain_vector(domains: Any, *, expected_length: int | None = None, name: str = "domains") -> np.ndarray:
    """Return a one-dimensional object array of hashable domain labels.

    Matrix-shaped inputs are interpreted row-wise so composite identifiers such as
    ``[[subject, run], ...]`` remain one domain label per sample instead of being
    flattened into unrelated scalar tokens.
    """
    array = np.asarray(domains, dtype=object)
    if array.ndim == 0:
        if expected_length is not None and int(expected_length) != 1:
            raise ValueError(f"{name} must contain {expected_length} rows")
        return _object_vector([_freeze_domain_label(array.item())])

    if expected_length is not None and int(array.shape[0]) != int(expected_length):
        raise ValueError(f"{name} must contain {expected_length} rows")

    if array.ndim == 1:
        return _object_vector(_freeze_domain_label(value) for value in array.tolist())
    if array.ndim == 2 and array.shape[1] == 1:
        return _object_vector(_freeze_domain_label(value) for value in array[:, 0].tolist())
    return _object_vector(_freeze_domain_label(row) for row in array.reshape(array.shape[0], -1).tolist())


def _unique_domain_labels(values: np.ndarray) -> tuple[Any, ...]:
    """Return first-occurrence unique domain labels from a domain vector."""
    unique: list[Any] = []
    seen: set[Any] = set()
    for value in values.tolist():
        frozen = _freeze_domain_label(value)
        if frozen not in seen:
            seen.add(frozen)
            unique.append(frozen)
    return tuple(unique)


def _domain_mask(values: np.ndarray, label: Any) -> np.ndarray:
    """Return a boolean mask for rows whose domain label equals ``label``."""
    frozen_label = _freeze_domain_label(label)
    return np.asarray([_freeze_domain_label(value) == frozen_label for value in values.tolist()], dtype=bool)


def _object_vector(values) -> np.ndarray:
    items = list(values)
    result = np.empty(len(items), dtype=object)
    result[:] = items
    return result


def _freeze_domain_label(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _freeze_domain_label(value.tolist())
    if isinstance(value, Mapping):
        items = ((_freeze_domain_label(key), _freeze_domain_label(item)) for key, item in value.items())
        return tuple(sorted(items, key=repr))
    if isinstance(value, Set) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(sorted((_freeze_domain_label(item) for item in value), key=repr))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_domain_label(item) for item in value)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value
