"""Treat composite tuple/list labels atomically in classifier helpers."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_classifier_tuple_labels_patch_installed"


def _object_vector(values: Sequence[Any]) -> np.ndarray:
    vector = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        vector[index] = value
    return vector


def _atomic_label_vector(labels: Sequence[Any] | np.ndarray) -> np.ndarray:
    """Return a 1-D label vector without expanding tuple/list labels.

    NumPy turns ``[("a", 1), ("b", 2)]`` into a two-dimensional array.  For
    classifier labels each row is one composite class value, so row-shaped input
    must be collapsed back into tuple objects before unique-class encoding.
    """

    if isinstance(labels, np.ndarray):
        if labels.ndim == 0:
            return _object_vector([labels.item()])
        if labels.ndim == 1:
            return labels.reshape(-1)
        rows = [tuple(row.tolist()) for row in labels.reshape(labels.shape[0], -1)]
        return _object_vector(rows)

    if isinstance(labels, (str, bytes)):
        return _object_vector([labels])

    try:
        items = list(labels)
    except TypeError:
        return _object_vector([labels])

    if any(isinstance(label, (tuple, list)) for label in items):
        return _object_vector([tuple(label) if isinstance(label, list) else label for label in items])

    array = np.asarray(items)
    if array.ndim <= 1:
        return array.reshape(-1)
    rows = [tuple(row.tolist()) for row in array.reshape(array.shape[0], -1)]
    return _object_vector(rows)


def _labels_equal(left: object, right: object) -> bool:
    try:
        equal = left == right
    except (TypeError, ValueError):
        return False
    try:
        return bool(equal)
    except (TypeError, ValueError):
        return False


def _label_mask(labels: np.ndarray, target: object) -> np.ndarray:
    return np.asarray([_labels_equal(label, target) for label in labels], dtype=bool)


def _unique_labels_and_inverse(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        classes, inverse = np.unique(labels, return_inverse=True)
        return classes, inverse.astype(int, copy=False)
    except (TypeError, ValueError):
        classes: list[object] = []
        encoded = np.empty(labels.shape[0], dtype=int)
        for row_index, label in enumerate(labels):
            for class_index, class_label in enumerate(classes):
                if _labels_equal(label, class_label):
                    encoded[row_index] = class_index
                    break
            else:
                encoded[row_index] = len(classes)
                classes.append(label)
        return _object_vector(classes), encoded


def install() -> None:
    classifiers = importlib.import_module("neureptrace.decoding.classifiers")
    if getattr(classifiers, _PATCH_MARKER, False):
        return

    def encode_classifier_labels(labels: Sequence[Any] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Encode labels as dense integer class ids while preserving composite labels."""

        label_vector = _atomic_label_vector(labels)
        if label_vector.size == 0:
            raise ValueError("At least one class label is required.")
        return _unique_labels_and_inverse(label_vector)

    original_fit = classifiers.CorrelationPrototypeClassifier.fit

    @wraps(original_fit)
    def fit(
        self,
        features: Sequence[Sequence[float]] | np.ndarray,
        labels: Sequence[Any] | np.ndarray,
        sample_weight: Sequence[float] | np.ndarray | None = None,
    ):
        features_array = np.asarray(features, dtype=float)
        label_vector = _atomic_label_vector(labels)
        if features_array.ndim != 2:
            raise ValueError("features must be a two-dimensional feature matrix.")
        if label_vector.shape[0] != features_array.shape[0]:
            raise ValueError("labels must contain one label per feature row.")

        classes, _encoded = _unique_labels_and_inverse(label_vector)
        if classes.size == 0:
            raise ValueError("At least one class is required.")
        self.classes_ = classes

        if sample_weight is None:
            self.prototypes_ = np.vstack(
                [np.mean(features_array[_label_mask(label_vector, class_label)], axis=0) for class_label in classes]
            )
        else:
            weights = np.asarray(sample_weight, dtype=float).reshape(-1)
            if weights.shape[0] != label_vector.shape[0]:
                raise ValueError("sample_weight must contain one weight per feature row.")
            if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
                raise ValueError("sample_weight must contain finite non-negative values.")
            self.prototypes_ = np.vstack(
                [
                    np.average(
                        features_array[_label_mask(label_vector, class_label)],
                        axis=0,
                        weights=weights[_label_mask(label_vector, class_label)],
                    )
                    for class_label in classes
                ]
            )

        self.normalized_prototypes_ = self._row_center_normalize(self.prototypes_)
        return self

    classifiers.encode_classifier_labels = encode_classifier_labels
    classifiers.CorrelationPrototypeClassifier.fit = fit
    setattr(classifiers, _PATCH_MARKER, True)
