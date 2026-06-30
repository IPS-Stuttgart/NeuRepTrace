"""Repair transfer null-class helpers for array-valued and composite labels."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from neureptrace._object_label_utils import replace_null_class_predictions as _replace_object_label_null_predictions

_INSTALLED = False
_ORIGINAL_APPEND_NULL_CLASS_FEATURES = None
_ORIGINAL_REPLACE_NULL_CLASS_PREDICTIONS = None
_OBJECT_DTYPE = np.dtype(object)


def _is_composite_label(value: object) -> bool:
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, np.ndarray):
        return value.ndim != 0
    return isinstance(value, (tuple, list, dict))


def _object_label_vector(length: int, label: object) -> np.ndarray:
    labels = np.empty(int(length), dtype=object)
    for index in range(labels.shape[0]):
        labels[index] = label
    return labels


def _constant_label_vector(length: int, label: object, dtype: np.dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    if dtype == _OBJECT_DTYPE or _is_composite_label(label):
        return _object_label_vector(length, label)
    try:
        return np.full(int(length), label, dtype=dtype)
    except (TypeError, ValueError, OverflowError):
        return _object_label_vector(length, label)


def install() -> None:
    """Install transfer null-label wrappers once."""

    global _INSTALLED, _ORIGINAL_APPEND_NULL_CLASS_FEATURES, _ORIGINAL_REPLACE_NULL_CLASS_PREDICTIONS
    if _INSTALLED:
        return

    from neureptrace.decoding import transfer

    _ORIGINAL_APPEND_NULL_CLASS_FEATURES = transfer.append_null_class_features
    _ORIGINAL_REPLACE_NULL_CLASS_PREDICTIONS = transfer.replace_null_class_predictions

    def _append_null_class_features(
        stimulus_features: Sequence[Sequence[float]] | np.ndarray,
        labels: Sequence | np.ndarray,
        null_features: Sequence[Sequence[float]] | np.ndarray | None = None,
        *,
        null_label: object = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Append null rows without unpacking composite null-label atoms."""

        stimulus_features_array = transfer._feature_matrix(stimulus_features, name="stimulus_features")
        label_vector = transfer._label_vector(labels, expected_length=stimulus_features_array.shape[0], name="labels")
        if null_features is None:
            return stimulus_features_array, label_vector

        null_features_array = transfer._feature_matrix(null_features, name="null_features")
        null_labels = _constant_label_vector(null_features_array.shape[0], null_label, label_vector.dtype)
        return np.vstack([stimulus_features_array, null_features_array]), np.concatenate([label_vector, null_labels])

    def _replace_null_class_predictions(
        predictions: Sequence | np.ndarray,
        *,
        null_label: object = 0,
        fallback_label: object = 1,
    ) -> np.ndarray:
        """Replace predicted null labels while preserving atomic composite labels."""

        return _replace_object_label_null_predictions(
            transfer._prediction_vector(predictions),
            null_label=null_label,
            fallback_label=fallback_label,
        )

    _append_null_class_features.__name__ = _ORIGINAL_APPEND_NULL_CLASS_FEATURES.__name__
    _append_null_class_features.__doc__ = _ORIGINAL_APPEND_NULL_CLASS_FEATURES.__doc__
    _replace_null_class_predictions.__name__ = _ORIGINAL_REPLACE_NULL_CLASS_PREDICTIONS.__name__
    _replace_null_class_predictions.__doc__ = _ORIGINAL_REPLACE_NULL_CLASS_PREDICTIONS.__doc__
    transfer.append_null_class_features = _append_null_class_features
    transfer.replace_null_class_predictions = _replace_null_class_predictions
    _INSTALLED = True
