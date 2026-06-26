"""Preserve composite source labels in adversarial torch decoders."""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Sequence
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_adversarial_composite_labels_patch_installed"


def _object_value_vector(values: Iterable[Any]) -> np.ndarray:
    items = list(values)
    vector = np.empty(len(items), dtype=object)
    for index, value in enumerate(items):
        vector[index] = value
    return vector


def _atomic_label_vector(values: Sequence[Any] | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    """Return one object-valued label per source row.

    NumPy converts rectangular inputs such as ``[(class_id, repetition), ...]``
    into a two-dimensional object array.  DANN/CDAN labels are row-level metadata,
    so each rectangular row must remain one atomic class value before integer
    encoding for torch.
    """

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
        raise ValueError(f"{name} must contain one label per source row: {vector.shape[0]} != {expected_length}.")
    return vector


def _values_equal(left: Any, right: Any) -> bool:
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


def _encode_atomic_labels(values: Sequence[Any] | np.ndarray, *, expected_length: int, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Encode atomic labels while preserving original class values."""

    vector = _atomic_label_vector(values, expected_length=expected_length, name=name)
    classes: list[Any] = []
    encoded = np.empty(vector.shape[0], dtype=np.int64)
    for index, label in enumerate(vector.tolist()):
        class_index = next((candidate for candidate, existing in enumerate(classes) if _values_equal(label, existing)), None)
        if class_index is None:
            class_index = len(classes)
            classes.append(label)
        encoded[index] = int(class_index)
    return _object_value_vector(classes), encoded


def _install_fit_wrapper(class_object: type, *, label_name: str) -> None:
    original_fit = class_object.fit
    if getattr(original_fit, _PATCH_MARKER, False):
        return

    @wraps(original_fit)
    def fit(self, source_features, source_labels, *, target_features):
        source = np.asarray(source_features)
        expected_length = int(source.shape[0]) if source.ndim >= 1 else 0
        classes, encoded_labels = _encode_atomic_labels(
            source_labels,
            expected_length=expected_length,
            name=label_name,
        )
        result = original_fit(self, source_features, encoded_labels, target_features=target_features)
        self.classes_ = classes
        return result

    setattr(fit, _PATCH_MARKER, True)
    class_object.fit = fit


def install() -> None:
    """Install composite-label wrappers for DANN and CDAN classifiers."""

    dann = importlib.import_module("neureptrace.decoding.dann")
    _install_fit_wrapper(dann.TorchDANNClassifier, label_name="DANN source_labels")

    cdan = importlib.import_module("neureptrace.decoding.cdan")
    _install_fit_wrapper(cdan.TorchCDANClassifier, label_name="CDAN source_labels")


__all__ = ["install"]
