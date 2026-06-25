# ruff: noqa
"""Composite-label support for windowed decoding helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from neureptrace._object_label_utils import label_accuracy, label_counts, values_equal

_INSTALLED = False
_ORIGINAL_LABEL_VECTOR = None


def _object_vector(values) -> np.ndarray:
    items = list(values)
    vector = np.empty(len(items), dtype=object)
    for index, value in enumerate(items):
        vector[index] = value
    return vector


def _is_composite_label(value: object) -> bool:
    if isinstance(value, np.ndarray):
        return value.ndim > 0
    return isinstance(value, (list, tuple))


def _as_atomic_label(value: object) -> object:
    if isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=object)
        if array.ndim == 0:
            return array.item()
        return tuple(array.reshape(-1).tolist())
    if isinstance(value, list):
        return tuple(value)
    return value


def _sequence_atomic_vector(values: Sequence | np.ndarray) -> np.ndarray | None:
    if isinstance(values, np.ndarray):
        return None
    items = list(values)
    if not items or not any(_is_composite_label(value) for value in items):
        return None
    return _object_vector(_as_atomic_label(value) for value in items)


def _label_vector(labels: Sequence | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    vector = _sequence_atomic_vector(labels)
    if vector is None:
        return _ORIGINAL_LABEL_VECTOR(labels, expected_length=expected_length, name=name)
    if vector.shape[0] != expected_length:
        raise ValueError(f"{name} length must match feature rows: {vector.shape[0]} != {expected_length}.")
    return vector


def _prediction_vector(predictions: Sequence | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    vector = _sequence_atomic_vector(predictions)
    if vector is None:
        try:
            array = np.asarray(predictions)
        except ValueError:
            vector = _object_vector(_as_atomic_label(value) for value in predictions)
        else:
            if array.ndim == 0:
                vector = np.asarray([array.item()])
            elif array.ndim == 1:
                vector = array.copy()
            elif 1 in array.shape:
                vector = array.reshape(-1).copy()
            elif array.dtype == object:
                rows = np.asarray(array, dtype=object).reshape(array.shape[0], -1)
                vector = _object_vector(tuple(row.tolist()) for row in rows)
            else:
                raise ValueError(f"{name} must be one-dimensional.")
    if vector.shape[0] != expected_length:
        raise ValueError(f"{name} length must match feature rows: {vector.shape[0]} != {expected_length}.")
    return vector


def _label_equal_mask(labels: Sequence | np.ndarray, label: object) -> np.ndarray:
    return np.asarray([values_equal(value, label) for value in labels], dtype=bool)


def _balanced_accuracy(predictions: Sequence | np.ndarray, labels: Sequence | np.ndarray) -> float:
    labels = _label_vector(labels, expected_length=len(labels), name="labels")
    predictions = _prediction_vector(predictions, expected_length=len(labels), name="predictions")
    if labels.size == 0:
        return np.nan
    label_values = label_counts(labels)[0]
    recalls = []
    for label in label_values:
        mask = _label_equal_mask(labels, label)
        if np.any(mask):
            recalls.append(label_accuracy(labels[mask], predictions[mask]))
    return float(np.mean(recalls)) if recalls else np.nan


def install() -> None:
    """Install the composite-label-safe windowed decoding patch."""

    global _INSTALLED, _ORIGINAL_LABEL_VECTOR
    if _INSTALLED:
        return

    from neureptrace.decoding import windowed

    _ORIGINAL_LABEL_VECTOR = windowed._label_vector
    windowed._label_vector = _label_vector
    windowed._balanced_accuracy = _balanced_accuracy

    original_predict_window_model = windowed.predict_window_model

    def predict_window_model(model_bundle: Any, features: Sequence[Sequence[float]] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        transformed_features = windowed.transform_window_features(model_bundle, features)
        predictions = _prediction_vector(
            model_bundle.model.predict(transformed_features),
            expected_length=transformed_features.shape[0],
            name="predictions",
        )
        scores = windowed.prediction_scores(model_bundle.model, transformed_features)
        return predictions, scores

    predict_window_model.__name__ = original_predict_window_model.__name__
    predict_window_model.__doc__ = original_predict_window_model.__doc__
    windowed.predict_window_model = predict_window_model
    _INSTALLED = True
