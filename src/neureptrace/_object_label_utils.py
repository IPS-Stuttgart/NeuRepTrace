"""Utilities for comparing and assigning composite object labels."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def values_equal(left: object, right: object) -> bool:
    """Compare labels without leaking tuple/list/array-valued equality."""

    if isinstance(left, np.generic):
        left = left.item()
    if isinstance(right, np.generic):
        right = right.item()

    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            left_array = np.asarray(left, dtype=object)
            right_array = np.asarray(right, dtype=object)
        except (TypeError, ValueError):
            return False
        if left_array.shape != right_array.shape:
            return False
        return all(
            values_equal(left_item, right_item)
            for left_item, right_item in zip(left_array.reshape(-1), right_array.reshape(-1), strict=True)
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

    array = np.asarray(values, dtype=object)
    return np.asarray([values_equal(value, label) for value in array], dtype=bool)


def assign_masked(array: np.ndarray, mask: np.ndarray, value: object) -> None:
    """Assign one possibly composite label value to each selected element."""

    if array.dtype == object:
        for index in np.flatnonzero(mask):
            array[index] = value
        return
    array[mask] = value


def _object_vector(values: Sequence[object]) -> np.ndarray:
    vector = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        vector[index] = value
    return vector


def label_counts(values: Sequence | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return stable unique labels and counts without sorting object labels."""

    unique: list[object] = []
    counts: list[int] = []
    for value in values:
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

    repaired = np.asarray(predictions).copy()
    null_mask = label_equal_mask(repaired, null_label)
    if not np.any(null_mask):
        return repaired
    non_null = repaired[~null_mask]
    if len(non_null) == 0:
        assign_masked(repaired, null_mask, fallback_label)
        return repaired
    nonzero_labels, counts = label_counts(non_null)
    assign_masked(repaired, null_mask, nonzero_labels[int(np.argmin(counts))])
    return repaired


def label_accuracy(labels: Sequence | np.ndarray, predictions: Sequence | np.ndarray) -> float:
    """Return mean equality for labels that may be composite objects."""

    if len(labels) == 0:
        return np.nan
    return float(np.mean([values_equal(label, prediction) for label, prediction in zip(labels, predictions, strict=True)]))
