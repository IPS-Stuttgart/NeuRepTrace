"""Preserve composite weak label-proportion class handling."""

from __future__ import annotations

from typing import Any

import numpy as np

import neureptrace.decoding.label_proportions as _label_proportions

_PATCH_MARKER = "_neureptrace_label_proportion_tuple_prediction_patch_installed"
_ORIGINAL_NORMALIZE_LABEL_PROPORTIONS = None


def _atomic_label(value: Any) -> Any:
    """Return a hashable scalar/composite class label."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=object)
        if array.ndim == 0:
            return _atomic_label(array.item())
        return tuple(_atomic_label(item) for item in array.reshape(-1).tolist())
    if isinstance(value, list):
        return tuple(_atomic_label(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_atomic_label(item) for item in value)
    return value


def _class_items(labels: Any) -> list[Any]:
    """Return one atomic class label per class column."""

    if isinstance(labels, np.ndarray):
        array = np.asarray(labels, dtype=object)
        if array.ndim == 0:
            return [_atomic_label(array.item())]
        if array.ndim == 1:
            return [_atomic_label(value) for value in array.tolist()]
        if array.ndim == 2:
            if array.shape[1] == 1 or array.shape[0] == 1:
                return [_atomic_label(value) for value in array.reshape(-1).tolist()]
            return [tuple(_atomic_label(item) for item in row.tolist()) for row in array]
        rows = array.reshape(array.shape[0], -1)
        return [tuple(_atomic_label(item) for item in row.tolist()) for row in rows]
    if isinstance(labels, (str, bytes)):
        return [labels]
    try:
        return [_atomic_label(value) for value in list(labels)]
    except TypeError:
        return [_atomic_label(labels)]


def _object_label_vector(labels: Any) -> np.ndarray:
    """Return a 1D object array without letting NumPy expand composite labels."""

    items = _class_items(labels)
    label_vector = np.empty(len(items), dtype=object)
    label_vector[:] = items
    return label_vector


def _normalize_label_proportions(target_proportions: Any, *, classes: Any = None) -> tuple[np.ndarray, tuple[Any, ...]]:
    """Normalize proportions after preserving composite class ids atomically."""

    class_order = None if classes is None else tuple(_object_label_vector(classes).tolist())
    return _ORIGINAL_NORMALIZE_LABEL_PROPORTIONS(target_proportions, classes=class_order)


def _predict_labels_from_label_proportions(result: _label_proportions.WeakLabelProportionCalibrationResult) -> np.ndarray:
    """Return argmax labels while treating composite class ids atomically."""

    class_vector = _object_label_vector(result.classes)
    return class_vector[np.argmax(result.probabilities, axis=1)]


def install() -> None:
    """Install tuple/composite-label-safe weak label-proportion helpers."""

    global _ORIGINAL_NORMALIZE_LABEL_PROPORTIONS
    if getattr(_label_proportions, _PATCH_MARKER, False):
        return
    _ORIGINAL_NORMALIZE_LABEL_PROPORTIONS = _label_proportions.normalize_label_proportions
    _label_proportions.normalize_label_proportions = _normalize_label_proportions
    _label_proportions.predict_labels_from_label_proportions = _predict_labels_from_label_proportions
    setattr(_label_proportions, _PATCH_MARKER, True)


__all__ = ["install"]
