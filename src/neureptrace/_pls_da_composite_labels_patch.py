"""Preserve composite row labels and reject complex PLS-DA features."""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Sequence
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_pls_da_composite_labels_patch_installed"
_TRANSFORM_PATCH_MARKER = "_neureptrace_pls_da_complex_transform_patch_installed"


def _object_value_vector(values: Iterable[Any]) -> np.ndarray:
    items = list(values)
    vector = np.empty(len(items), dtype=object)
    for index, value in enumerate(items):
        vector[index] = value
    return vector


def _labels_equal(left: Any, right: Any) -> bool:
    try:
        result = left == right
    except Exception:  # pragma: no cover - defensive for unusual label objects
        return False
    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    try:
        return bool(np.all(result))
    except Exception:  # pragma: no cover - defensive for unusual label objects
        return False


def _contains_complex_feature(value: object) -> bool:
    """Return whether a feature container contains complex-valued entries."""

    if isinstance(value, (complex, np.complexfloating)):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.complexfloating):
            return bool(value.size)
        if value.dtype == object:
            return any(_contains_complex_feature(item) for item in value.ravel(order="C"))
        return False
    if hasattr(value, "__array__"):
        try:
            return _contains_complex_feature(np.asarray(value))
        except (TypeError, ValueError):
            return False
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, Sequence):
        return any(_contains_complex_feature(item) for item in value)
    return False


def _validate_real_features(features: object) -> None:
    if _contains_complex_feature(features):
        raise ValueError(
            "PLSDiscriminantTransformer features must contain real-valued values, not complex values."
        )


def _atomic_label_vector(values: Sequence[Any] | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    """Return one object-valued class label per feature row."""

    array = np.asarray(values, dtype=object)
    if array.ndim == 0:
        vector = _object_value_vector([array.item()])
    elif array.ndim == 1:
        if array.shape[0] == expected_length:
            vector = _object_value_vector(array.tolist())
        elif expected_length == 1:
            vector = _object_value_vector([tuple(array.tolist())])
        else:
            vector = _object_value_vector(array.reshape(-1).tolist())
    else:
        rows = array.reshape(array.shape[0], -1)
        if rows.shape[1] == 1:
            vector = _object_value_vector(rows[:, 0].tolist())
        else:
            vector = _object_value_vector(tuple(row.tolist()) for row in rows)

    if vector.shape[0] != expected_length:
        raise ValueError(f"{name} must contain one label per feature row: {vector.shape[0]} != {expected_length}.")
    return vector


def _has_rectangular_row_labels(values: Sequence[Any] | np.ndarray, *, expected_length: int) -> bool:
    array = np.asarray(values, dtype=object)
    if array.ndim < 2 or array.shape[0] != expected_length:
        return False
    rows = array.reshape(array.shape[0], -1)
    return rows.shape[1] > 1


def _encode_atomic_labels(values: Sequence[Any] | np.ndarray, *, expected_length: int, name: str) -> tuple[np.ndarray, np.ndarray]:
    vector = _atomic_label_vector(values, expected_length=expected_length, name=name)
    classes: list[Any] = []
    encoded = np.empty(vector.shape[0], dtype=np.int64)
    for index, label in enumerate(vector.tolist()):
        class_index = next((candidate for candidate, existing in enumerate(classes) if _labels_equal(label, existing)), None)
        if class_index is None:
            class_index = len(classes)
            classes.append(label)
        encoded[index] = int(class_index)
    return _object_value_vector(classes), encoded


def install() -> None:
    """Patch PLS-DA label encoding and reject complex feature values."""

    decoding = importlib.import_module("neureptrace.decoding")
    cls = decoding.PLSDiscriminantTransformer
    original_fit = cls.fit
    if not getattr(original_fit, _PATCH_MARKER, False):

        @wraps(original_fit)
        def fit(self, features: Sequence[Sequence[float]] | np.ndarray, labels: Sequence[Any] | np.ndarray):
            _validate_real_features(features)
            feature_array = np.asarray(features)
            expected_length = int(feature_array.shape[0]) if feature_array.ndim >= 1 else 0
            if not _has_rectangular_row_labels(labels, expected_length=expected_length):
                return original_fit(self, features, labels)

            classes, encoded = _encode_atomic_labels(
                labels,
                expected_length=expected_length,
                name="PLSDiscriminantTransformer labels",
            )
            result = original_fit(self, features, encoded)
            self.classes_ = classes
            return result

        setattr(fit, _PATCH_MARKER, True)
        cls.fit = fit

    original_transform = cls.transform
    if not getattr(original_transform, _TRANSFORM_PATCH_MARKER, False):

        @wraps(original_transform)
        def transform(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
            _validate_real_features(features)
            return original_transform(self, features)

        setattr(transform, _TRANSFORM_PATCH_MARKER, True)
        cls.transform = transform


__all__ = ["install"]
