"""Preserve composite labels in supervised PLS low-rank features."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_supervised_pls_tuple_labels_patch_installed"


def _as_label_vector(labels: Any, *, expected_length: int) -> np.ndarray:
    if isinstance(labels, np.ndarray):
        if labels.ndim == 0:
            raise ValueError("SupervisedPLSTransformer expects one label per training row.")
        if labels.ndim == 1:
            values = labels.tolist()
        else:
            array = np.asarray(labels, dtype=object)
            values = [_sequence_label_from_row(row) for row in array.reshape(array.shape[0], -1)]
    else:
        try:
            values = list(labels)
        except TypeError as exc:
            raise ValueError("SupervisedPLSTransformer expects one label per training row.") from exc
    if len(values) != expected_length:
        raise ValueError(f"SupervisedPLSTransformer expects one label per training row: {len(values)} != {expected_length}.")
    vector = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        vector[index] = _hashable_label(value)
    return vector


def _sequence_label_from_row(row: np.ndarray) -> Any:
    flat = np.asarray(row, dtype=object).reshape(-1)
    if flat.size == 1:
        return _hashable_label(flat[0])
    return tuple(_hashable_label(value) for value in flat.tolist())


def _hashable_label(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _hashable_label(value.item())
    if isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=object)
        if array.ndim == 0:
            return _hashable_label(array.item())
        return tuple(_hashable_label(item) for item in array.reshape(-1).tolist())
    if isinstance(value, list):
        return tuple(_hashable_label(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_hashable_label(item) for item in value)
    if isinstance(value, dict):
        return tuple((_hashable_label(key), _hashable_label(item)) for key, item in sorted(value.items(), key=_dict_item_sort_key))
    return value


def _dict_item_sort_key(item: tuple[Any, Any]) -> tuple[str, str, str]:
    key, _value = item
    return (type(key).__module__, type(key).__qualname__, repr(key))


def _label_key(label: Any) -> tuple[Any, ...]:
    label = _hashable_label(label)
    if isinstance(label, float) and np.isnan(label):
        return ("nan",)
    if isinstance(label, bytes):
        return ("bytes", label)
    if isinstance(label, str):
        return ("str", label)
    if isinstance(label, (bool, np.bool_)):
        return ("bool", bool(label))
    if isinstance(label, tuple):
        return ("sequence", tuple(_label_key(value) for value in label))
    try:
        hash(label)
    except TypeError:
        return ("repr", type(label).__module__, type(label).__qualname__, repr(label))
    return ("scalar", label)


def _unique_label_vector(labels: np.ndarray) -> np.ndarray:
    unique: list[Any] = []
    seen: set[tuple[Any, ...]] = set()
    for label in labels.tolist():
        key = _label_key(label)
        if key not in seen:
            seen.add(key)
            unique.append(label)
    vector = np.empty(len(unique), dtype=object)
    for index, label in enumerate(unique):
        vector[index] = label
    return vector


def _encode_labels(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    class_to_index = {_label_key(label): index for index, label in enumerate(classes.tolist())}
    return np.asarray([class_to_index[_label_key(label)] for label in labels.tolist()], dtype=int)


def install() -> None:
    """Patch supervised PLS label handling for composite class labels."""

    lowrank = importlib.import_module("neureptrace.bushmeg_supervised_lowrank_loso")
    original_fit = lowrank.SupervisedPLSTransformer.fit
    if getattr(original_fit, _PATCH_MARKER, False):
        return

    @wraps(original_fit)
    def fit(self, features, labels):
        features = np.asarray(features, dtype=float)
        if features.ndim != 2:
            raise ValueError("SupervisedPLSTransformer expects a two-dimensional feature matrix.")
        label_vector = _as_label_vector(labels, expected_length=features.shape[0])
        classes = _unique_label_vector(label_vector)
        if classes.size < 2:
            raise ValueError("SupervisedPLSTransformer needs at least two classes.")
        requested = lowrank._normalize_pls_components(self.n_components)
        feasible = min(requested, features.shape[0] - 1, features.shape[1])
        if feasible < 1:
            raise ValueError(
                "Supervised PLS needs at least two training examples and one input feature; "
                f"got shape {features.shape}."
            )
        targets = np.eye(classes.size, dtype=float)[_encode_labels(label_vector, classes)]
        self.classes_ = classes
        self.n_components_ = int(feasible)
        self.pls_ = lowrank.PLSRegression(n_components=self.n_components_, scale=bool(self.scale))
        self.pls_.fit(features, targets)
        return self

    setattr(fit, _PATCH_MARKER, True)
    lowrank.SupervisedPLSTransformer.fit = fit


__all__ = ["install"]
