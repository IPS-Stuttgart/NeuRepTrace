"""Utilities for comparing and assigning composite object labels."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _numpy_nat_key(value: object) -> tuple[str, str] | None:
    """Return a stable key for NumPy temporal NaT scalars without coercing to None."""

    if isinstance(value, (np.datetime64, np.timedelta64)) and bool(np.isnat(value)):
        return (type(value).__module__, type(value).__qualname__)
    return None


def _missing_scalar_key(value: object) -> tuple[str, str] | None:
    """Return a stable key for scalar missing-value sentinels."""

    numpy_nat_key = _numpy_nat_key(value)
    if numpy_nat_key is not None:
        return numpy_nat_key
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (np.ndarray, list, tuple, dict)):
        return None
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return None
    if not isinstance(missing, (bool, np.bool_)) or not bool(missing):
        return None
    return (type(value).__module__, type(value).__qualname__)


def _both_nan(left: object, right: object) -> bool:
    """Return whether both scalar labels are the same missing-value sentinel."""

    left_key = _missing_scalar_key(left)
    return left_key is not None and left_key == _missing_scalar_key(right)


def _comparable_scalar(value: object) -> object:
    """Return a scalar value suitable for equality while preserving NumPy NaT."""

    if _numpy_nat_key(value) is not None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    return value


def _array_for_comparison(value: object) -> np.ndarray:
    """Return an array view without converting temporal NaT scalars to None."""

    if isinstance(value, np.ndarray):
        return np.asarray(value)
    return np.asarray(value, dtype=object)


def _array_items(array: np.ndarray) -> list[object]:
    """Return flattened array items while preserving NumPy scalar labels."""

    if array.ndim == 0:
        return [array[()]]
    flat = array.reshape(-1)
    return [flat[index] for index in range(flat.shape[0])]


def _label_items(values: Sequence | np.ndarray) -> list[object]:
    """Return top-level label atoms while preserving NumPy temporal scalars."""

    if isinstance(values, (str, bytes)):
        return [values]
    if isinstance(values, np.ndarray):
        array = np.asarray(values)
        if array.ndim == 0:
            return [array[()]]
        return [array[index] for index in range(array.shape[0])]
    return list(values)


def values_equal(left: object, right: object) -> bool:
    """Compare labels without leaking tuple/list/array-valued equality."""

    if _both_nan(left, right):
        return True

    left = _comparable_scalar(left)
    right = _comparable_scalar(right)

    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            left_array = _array_for_comparison(left)
            right_array = _array_for_comparison(right)
        except (TypeError, ValueError):
            return False
        if left_array.shape != right_array.shape:
            return False
        return all(
            values_equal(left_item, right_item)
            for left_item, right_item in zip(_array_items(left_array), _array_items(right_array), strict=True)
        )

    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
            return False
        if len(left) != len(right):
            return False
        return all(values_equal(left_item, right_item) for left_item, right_item in zip(left, right, strict=True))

    try:
        equal = left == right
    except (TypeError, ValueError):
        return False
    if isinstance(equal, (bool, np.bool_)):
        return bool(equal)
    try:
        return bool(np.all(equal))
    except (TypeError, ValueError):
        return False


def label_equal_mask(values: Sequence | np.ndarray, label: object) -> np.ndarray:
    """Return an equality mask for arbitrary scalar or composite labels."""

    return np.asarray([values_equal(value, label) for value in _label_items(values)], dtype=bool)


def assign_masked(array: np.ndarray, mask: np.ndarray, value: object) -> None:
    """Assign one possibly composite label value to each selected element."""

    mask = np.asarray(mask, dtype=bool)
    if array.dtype == object:
        if mask.shape == array.shape:
            flat_values = array.reshape(-1)
            for index in np.flatnonzero(mask.reshape(-1)):
                flat_values[index] = value
        else:
            array[mask] = value
        return
    array[mask] = value


def _object_vector(values: Sequence[object]) -> np.ndarray:
    vector = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        vector[index] = value
    return vector


def _object_array_copy(values: np.ndarray) -> np.ndarray:
    object_values = np.empty(values.shape, dtype=object)
    object_values[...] = values
    return object_values


def _ensure_assignable(array: np.ndarray, mask: np.ndarray, value: object) -> np.ndarray:
    """Promote to object dtype when NumPy cannot store the replacement label."""

    if array.dtype == object:
        return array
    try:
        trial = array.copy()
        trial[np.asarray(mask, dtype=bool)] = value
    except (TypeError, ValueError, OverflowError):
        return _object_array_copy(array)
    return array


def label_counts(values: Sequence | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return stable unique labels and counts without sorting object labels."""

    unique: list[object] = []
    counts: list[int] = []
    for value in _label_items(values):
        for index, existing in enumerate(unique):
            if values_equal(value, existing):
                counts[index] += 1
                break
        else:
            unique.append(value)
            counts.append(1)
    return _object_vector(unique), np.asarray(counts, dtype=int)


def replace_null_class_predictions(predictions: Sequence | np.ndarray, *, null_label: object = 0, fallback_label: object = 1) -> np.ndarray:
    """Replace null predictions without broadcasting composite fallback labels."""

    if isinstance(predictions, np.ndarray) and predictions.dtype != object:
        repaired = predictions.copy()
    else:
        repaired = _object_vector(list(predictions))
    null_mask = label_equal_mask(repaired, null_label)
    if not np.any(null_mask):
        return repaired
    non_null = repaired[~null_mask]
    if len(non_null) == 0:
        repaired = _ensure_assignable(repaired, null_mask, fallback_label)
        assign_masked(repaired, null_mask, fallback_label)
        return repaired
    nonzero_labels, counts = label_counts(non_null)
    replacement = nonzero_labels[int(np.argmin(counts))]
    repaired = _ensure_assignable(repaired, null_mask, replacement)
    assign_masked(repaired, null_mask, replacement)
    return repaired


def label_accuracy(labels: Sequence | np.ndarray, predictions: Sequence | np.ndarray) -> float:
    """Return mean equality for labels that may be composite objects."""

    if len(labels) == 0:
        return np.nan
    return float(np.mean([values_equal(label, prediction) for label, prediction in zip(labels, predictions, strict=True)]))
