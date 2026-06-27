"""Preserve tuple-valued domains and anchors in unlabeled anchor alignment."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_unlabeled_anchor_tuple_patch_installed"


class _AtomicObjectVector(np.ndarray):
    """Object vector whose equality keeps tuple values atomic."""

    def __eq__(self, other: object) -> np.ndarray:  # type: ignore[override]
        return np.asarray([_values_equal(value, other) for value in np.asarray(self, dtype=object).reshape(-1).tolist()], dtype=bool)

    def __ne__(self, other: object) -> np.ndarray:  # type: ignore[override]
        return np.logical_not(self.__eq__(other))


def _values_equal(left: object, right: object) -> bool:
    try:
        result = left == right
    except (TypeError, ValueError):
        return False
    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    try:
        return bool(np.all(result))
    except (TypeError, ValueError):
        return False


def _object_vector(items: list[Any]) -> np.ndarray:
    vector = np.empty(len(items), dtype=object)
    for index, item in enumerate(items):
        vector[index] = item
    return vector.view(_AtomicObjectVector)


def _vector_items(values: Any, *, expected_length: int, name: str) -> list[Any]:
    array = np.asarray(values, dtype=object)
    if array.ndim == 0:
        items = [array.item()]
    elif array.ndim == 1:
        if array.shape[0] == expected_length:
            items = array.tolist()
        elif expected_length == 1:
            items = [tuple(array.tolist())]
        else:
            items = array.reshape(-1).tolist()
    elif 1 in array.shape and array.size == expected_length:
        items = array.reshape(-1).tolist()
    elif array.shape[0] == expected_length:
        rows = array.reshape(array.shape[0], -1)
        if rows.shape[1] == 1:
            items = rows[:, 0].tolist()
        else:
            items = [tuple(row.tolist()) for row in rows]
    else:
        raise ValueError(f"{name} must contain one value per row or one composite row per sample.")
    return items


def install() -> None:
    """Install tuple-safe vector normalization for unlabeled anchor alignment."""

    module = importlib.import_module("neureptrace.decoding.unlabeled_anchor_alignment")
    if getattr(module, _PATCH_MARKER, False):
        return

    def _hashable_vector(values: Any, *, expected_length: int, name: str, allow_missing: bool = False) -> np.ndarray:
        vector = _object_vector(_vector_items(values, expected_length=expected_length, name=name))
        if vector.shape[0] != expected_length:
            raise ValueError(f"{name} must contain one value per row: {vector.shape[0]} != {expected_length}.")
        for value in vector.tolist():
            if allow_missing and module._is_missing_anchor(value):
                continue
            try:
                hash(value)
            except TypeError as exc:
                raise ValueError(f"{name} values must be hashable; got {value!r}.") from exc
        return vector

    module._hashable_vector = _hashable_vector
    setattr(module, _PATCH_MARKER, True)


__all__ = ["install"]
