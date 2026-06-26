from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from neureptrace.decoding.windowed import WindowedModelBundle, transform_window_features


def _object_label_vector(labels: Sequence[object]) -> np.ndarray:
    vector = np.empty(len(labels), dtype=object)
    for index, label in enumerate(labels):
        vector[index] = label
    return vector


def _label_vector(labels: Sequence | np.ndarray) -> np.ndarray:
    """Return a one-dimensional class-label vector without splitting tuples."""

    if isinstance(labels, np.ndarray):
        if labels.ndim == 0:
            return labels.reshape(1)
        if labels.dtype == object and labels.ndim == 1:
            return labels
        return labels.ravel()

    if isinstance(labels, (str, bytes)):
        return np.asarray([labels])

    try:
        items = list(labels)
    except TypeError:
        return np.asarray([labels])

    if any(isinstance(label, tuple) for label in items):
        return _object_label_vector(items)
    return np.asarray(items).ravel()


def _labels_equal(left: object, right: object) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return bool(np.array_equal(left, right))
    try:
        equal = left == right
    except (TypeError, ValueError):
        return False
    try:
        return bool(equal)
    except (TypeError, ValueError):
        return False


def _label_matches(classes: np.ndarray, target: object) -> np.ndarray:
    return np.asarray([_labels_equal(label, target) for label in _label_vector(classes)], dtype=bool)


def _unique_label_vector(labels: Sequence | np.ndarray) -> np.ndarray:
    vector = _label_vector(labels)
    if vector.dtype != object:
        return np.unique(vector)

    unique: list[object] = []
    for label in vector:
        if not any(_labels_equal(label, existing) for existing in unique):
            unique.append(label)
    return _object_label_vector(unique)


def model_classes(model: Any, fallback_labels: Sequence | np.ndarray | None = None) -> np.ndarray | None:
    """Return fitted class labels from a classifier or sklearn-style pipeline."""

    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            classes = getattr(step, "classes_", None)
            if classes is not None:
                break
    if classes is None and fallback_labels is not None:
        classes = _unique_label_vector(fallback_labels)
    if classes is None:
        return None
    return _label_vector(classes)


def as_class_score_matrix(
    raw_scores: Sequence[Sequence[float]] | Sequence[float] | np.ndarray,
    classes: Sequence | np.ndarray,
    *,
    n_samples: int,
) -> np.ndarray | None:
    """Normalize classifier score output to ``(n_samples, n_classes)``.

    Binary ``decision_function`` outputs are expanded so column 0 scores the
    first class and column 1 scores the second class, matching sklearn's
    ``classes_`` convention for linear binary decision scores.
    """

    classes = _label_vector(classes)
    scores = np.asarray(raw_scores, dtype=float)
    if scores.ndim == 1:
        if scores.shape[0] != n_samples or classes.size != 2:
            return None
        return np.column_stack((-scores, scores))
    if scores.ndim != 2 or scores.shape[0] != n_samples:
        return None
    if scores.shape[1] == classes.size:
        return scores
    if scores.shape[1] == 1 and classes.size == 2:
        column = scores[:, 0]
        return np.column_stack((-column, column))
    return None


def _prediction_class_score_matrix(predictions: Sequence | np.ndarray, classes: np.ndarray, *, n_samples: int) -> np.ndarray | None:
    predictions = _label_vector(predictions)
    classes = _label_vector(classes)
    if predictions.shape[0] != n_samples or classes.size == 0:
        return None
    scores = np.zeros((n_samples, classes.size), dtype=float)
    for row_index, predicted in enumerate(predictions):
        matches = np.flatnonzero(_label_matches(classes, predicted))
        if matches.size:
            scores[row_index, matches[0]] = 1.0
    return scores


def class_score_matrix(
    model: Any,
    features: Sequence[Sequence[float]] | np.ndarray,
    *,
    classes: Sequence | np.ndarray | None = None,
    fallback_labels: Sequence | np.ndarray | None = None,
    score_methods: Sequence[str] = ("decision_function", "predict_proba"),
    predict_fallback: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return per-class scores and the corresponding class order.

    The helper tries common classifier scoring APIs in order, normalizes binary
    decision scores to two class columns, and can optionally fall back to a
    one-hot prediction score matrix when a classifier has no scoring API.
    """

    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError("features must be a two-dimensional feature matrix.")

    class_order = _label_vector(classes) if classes is not None else model_classes(model, fallback_labels=fallback_labels)
    if class_order is None or class_order.size == 0:
        return None, None

    for method_name in score_methods:
        if not hasattr(model, method_name):
            continue
        scores = as_class_score_matrix(getattr(model, method_name)(features), class_order, n_samples=features.shape[0])
        if scores is not None:
            return scores, class_order
    if predict_fallback and hasattr(model, "predict"):
        scores = _prediction_class_score_matrix(model.predict(features), class_order, n_samples=features.shape[0])
        if scores is not None:
            return scores, class_order
    return None, None


def predict_window_class_scores(
    model_bundle: WindowedModelBundle,
    features: Sequence[Sequence[float]] | np.ndarray,
    *,
    predict_fallback: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return a windowed model's per-class score matrix and class order."""

    transformed_features = transform_window_features(model_bundle, features)
    return class_score_matrix(
        model_bundle.model,
        transformed_features,
        fallback_labels=model_bundle.train_labels,
        predict_fallback=predict_fallback,
    )
