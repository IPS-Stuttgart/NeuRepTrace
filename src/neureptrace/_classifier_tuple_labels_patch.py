"""Classifier label compatibility patches."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from functools import wraps
from typing import Any

import numpy as np

from neureptrace._object_label_utils import values_equal

_PATCH_MARKER = "_neureptrace_classifier_tuple_labels_patch_installed"


def _object_vector(values: Sequence[Any]) -> np.ndarray:
    out = np.empty(len(values), dtype=object)
    for idx, value in enumerate(values):
        out[idx] = value
    return out


def _atomic_label_vector(labels: Sequence[Any] | np.ndarray) -> np.ndarray:
    if isinstance(labels, np.ndarray):
        if labels.ndim == 0:
            return _object_vector([labels.item()])
        if labels.ndim == 1:
            return labels.reshape(-1)
        return _object_vector([tuple(row.tolist()) for row in labels.reshape(labels.shape[0], -1)])
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
    return _object_vector([tuple(row.tolist()) for row in array.reshape(array.shape[0], -1)])


def _labels_equal(left: object, right: object) -> bool:
    return values_equal(left, right)


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


def _as_python_scalar(value: object) -> object:
    return value.item() if isinstance(value, np.generic) else value


def _validate_sample_weights(*, n_samples: int, class_labels: np.ndarray, class_masks: Sequence[np.ndarray], sample_weight: Sequence[float] | np.ndarray) -> np.ndarray:
    weights = np.asarray(sample_weight, dtype=float).reshape(-1)
    if weights.shape[0] != n_samples:
        raise ValueError("sample_weight must contain one weight per feature row.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("sample_weight must contain finite non-negative values.")
    zero_weight_classes = [_as_python_scalar(label) for label, mask in zip(class_labels, class_masks, strict=True) if float(weights[mask].sum()) <= 0.0]
    if zero_weight_classes:
        raise ValueError("sample_weight must assign positive total weight to every class; zero-weight classes: " f"{zero_weight_classes!r}.")
    return weights


def install() -> None:
    classifiers = importlib.import_module("neureptrace.decoding.classifiers")
    if getattr(classifiers, _PATCH_MARKER, False):
        return

    def encode_classifier_labels(labels: Sequence[Any] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        label_vector = _atomic_label_vector(labels)
        if label_vector.size == 0:
            raise ValueError("At least one class label is required.")
        return _unique_labels_and_inverse(label_vector)

    original_fit = classifiers.CorrelationPrototypeClassifier.fit
    original_decision_function = classifiers.DecodedLabelClassifier.decision_function

    @wraps(original_fit)
    def fit(self, features: Sequence[Sequence[float]] | np.ndarray, labels: Sequence[Any] | np.ndarray, sample_weight: Sequence[float] | np.ndarray | None = None):
        features_array = np.asarray(features, dtype=float)
        label_vector = _atomic_label_vector(labels)
        if features_array.ndim != 2:
            raise ValueError("features must be a two-dimensional feature matrix.")
        if label_vector.shape[0] != features_array.shape[0]:
            raise ValueError("labels must contain one label per feature row.")
        classes, _ = _unique_labels_and_inverse(label_vector)
        if classes.size == 0:
            raise ValueError("At least one class is required.")
        self.classes_ = classes
        class_masks = [_label_mask(label_vector, class_label) for class_label in classes]
        if sample_weight is None:
            self.prototypes_ = np.vstack([np.mean(features_array[mask], axis=0) for mask in class_masks])
        else:
            weights = _validate_sample_weights(n_samples=label_vector.shape[0], class_labels=classes, class_masks=class_masks, sample_weight=sample_weight)
            self.prototypes_ = np.vstack([np.average(features_array[mask], axis=0, weights=weights[mask]) for mask in class_masks])
        self.normalized_prototypes_ = self._row_center_normalize(self.prototypes_)
        return self

    @wraps(original_decision_function)
    def decision_function(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        if hasattr(self.model, "decision_function"):
            scores = np.asarray(self.model.decision_function(features), dtype=float)
            if scores.ndim == 1 and self.classes_.shape[0] == 2:
                return np.column_stack((-0.5 * scores, 0.5 * scores))
            return scores
        return original_decision_function(self, features)

    classifiers.encode_classifier_labels = encode_classifier_labels
    classifiers.CorrelationPrototypeClassifier.fit = fit
    classifiers.DecodedLabelClassifier.decision_function = decision_function
    setattr(classifiers, _PATCH_MARKER, True)
