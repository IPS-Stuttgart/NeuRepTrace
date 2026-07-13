"""Preserve NumPy composite labels and validate class-limit controls."""

from __future__ import annotations

import numpy as np

from neureptrace._object_label_utils import values_equal

_PATCH_MARKER = "_neureptrace_sampling_composite_label_array_patch_installed"
_INTEGER_PATCH_MARKER = "_neureptrace_sampling_integer_array_patch_installed"


def _coerce_label_item(item):
    """Keep NumPy array-valued labels atomic when labels are supplied as a sequence."""

    if not isinstance(item, np.ndarray):
        return item
    array = item.astype(object, copy=False)
    if array.ndim == 0:
        return array.item()
    flat = array.reshape(-1)
    if flat.size == 1:
        return flat[0]
    return tuple(flat.tolist())


def _array_label_vector(array: np.ndarray) -> np.ndarray:
    """Return one class-label object per row of an array-like label container."""

    array = array.astype(object, copy=False)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim == 1:
        return array
    rows = array.reshape(array.shape[0], -1)
    if rows.shape[1] == 1:
        return rows[:, 0].astype(object, copy=False)
    vector = np.empty(rows.shape[0], dtype=object)
    vector[:] = [tuple(row.tolist()) for row in rows]
    return vector


def _label_vector(labels) -> np.ndarray:
    """Return labels as a one-dimensional object vector without splitting rows."""

    if isinstance(labels, np.ndarray):
        return _array_label_vector(labels)

    if isinstance(labels, (str, bytes)):
        return np.asarray([labels], dtype=object)

    try:
        items = list(labels)
    except TypeError:
        items = [labels]

    items = [_coerce_label_item(item) for item in items]
    if any(isinstance(item, tuple) for item in items):
        vector = np.empty(len(items), dtype=object)
        vector[:] = items
        return vector

    array = np.asarray(items, dtype=object)
    if array.ndim >= 2 and array.shape[0] == len(items):
        return _array_label_vector(array)
    return array.reshape(-1)


def _labels_equal(left: object, right: object) -> bool:
    return values_equal(left, right)


def _normalize_integer(value, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"{name} must be an integer.")
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not np.isfinite(number) or number % 1.0 != 0.0:
        raise ValueError(f"{name} must be an integer.")
    integer = int(number)
    if minimum is not None and integer < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return integer


def install() -> None:
    """Install sampler fixes for labels and integer-like controls."""

    from neureptrace.decoding import sampling

    if not getattr(sampling, _PATCH_MARKER, False):
        sampling._label_vector = _label_vector
        sampling._labels_equal = _labels_equal
        setattr(sampling, _PATCH_MARKER, True)

    if not getattr(sampling, _INTEGER_PATCH_MARKER, False):
        sampling._normalize_integer = _normalize_integer
        setattr(sampling, _INTEGER_PATCH_MARKER, True)


__all__ = ["install"]
