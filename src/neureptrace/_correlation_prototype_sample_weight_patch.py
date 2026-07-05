"""Runtime patch for weighted correlation-prototype class-support validation."""

from __future__ import annotations

from collections.abc import Sequence
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_correlation_prototype_sample_weight_patch_installed"


def _object_vector(values: Sequence[Any]) -> np.ndarray:
    out = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        out[index] = value
    return out


def _atomic_label_vector(labels: Sequence[Any] | np.ndarray) -> np.ndarray:
    """Return one atomic label per sample without flattening composite labels."""

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


def _values_equal(left: object, right: object) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return bool(np.array_equal(left, right))
        except Exception:
            return False
    try:
        comparison = left == right
    except (TypeError, ValueError):
        return False
    if isinstance(comparison, np.ndarray):
        if comparison.shape == ():
            return bool(comparison.item())
        return bool(np.all(comparison))
    return bool(comparison)


def _stable_unique_labels(labels: np.ndarray) -> np.ndarray:
    classes: list[Any] = []
    for label in labels:
        if not any(_values_equal(label, class_label) for class_label in classes):
            classes.append(label)
    return _object_vector(classes)


def _label_mask(labels: np.ndarray, target: object) -> np.ndarray:
    return np.asarray([_values_equal(label, target) for label in labels], dtype=bool)


def _as_python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _zero_weight_classes(labels: Sequence[object] | np.ndarray, sample_weight: Sequence[float] | np.ndarray) -> list[Any]:
    labels_array = _atomic_label_vector(labels)
    try:
        weights = np.asarray(sample_weight, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return []
    if weights.shape[0] != labels_array.shape[0]:
        return []
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        return []

    return [
        _as_python_scalar(class_label)
        for class_label in _stable_unique_labels(labels_array)
        if float(weights[_label_mask(labels_array, class_label)].sum()) <= 0.0
    ]


def install() -> None:
    """Reject weighted prototype fits that leave any class without support."""

    from neureptrace.decoding import classifiers

    classifier_type = classifiers.CorrelationPrototypeClassifier
    if getattr(classifier_type.fit, _PATCH_MARKER, False):
        return

    original_fit = classifier_type.fit

    @wraps(original_fit)
    def fit(
        self,
        features: Sequence[Sequence[float]] | np.ndarray,
        labels: Sequence[object] | np.ndarray,
        sample_weight: Sequence[float] | np.ndarray | None = None,
    ):
        if sample_weight is not None:
            unsupported_classes = _zero_weight_classes(labels, sample_weight)
            if unsupported_classes:
                raise ValueError(
                    "sample_weight must assign positive total weight to every class; "
                    f"zero-weight classes: {unsupported_classes!r}."
                )
        return original_fit(self, features, labels, sample_weight=sample_weight)

    setattr(fit, _PATCH_MARKER, True)
    classifier_type.fit = fit


__all__ = ["install"]
