"""Preserve composite labels/domains in source-domain generalization encoders."""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import numpy as np
import pandas as pd

from . import _source_ensemble_tuple_domains_patch
from ._object_label_utils import values_equal as _values_equal

_PATCH_MARKER = "_neureptrace_source_domain_generalization_composite_patch_installed"


def _object_value_vector(values: Sequence[Any]) -> np.ndarray:
    vector = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        vector[index] = value
    return vector


def _materialize_iterators(value: Any) -> Any:
    """Recursively replace one-pass iterators with stable tuple values."""

    if isinstance(value, np.ndarray):
        if value.dtype != object:
            return value
        materialized = np.empty(value.shape, dtype=object)
        for index in np.ndindex(value.shape):
            materialized[index] = _materialize_iterators(value[index])
        return materialized
    if isinstance(value, Iterator):
        return tuple(_materialize_iterators(item) for item in value)
    if isinstance(value, list):
        return [_materialize_iterators(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_materialize_iterators(item) for item in value)
    if isinstance(value, dict):
        return {
            _materialize_iterators(key): _materialize_iterators(item)
            for key, item in value.items()
        }
    if isinstance(value, set):
        return {_materialize_iterators(item) for item in value}
    if isinstance(value, frozenset):
        return frozenset(_materialize_iterators(item) for item in value)
    return value


def _atomic_vector(values: Iterable[Any] | np.ndarray, *, name: str, reject_matrix: bool = False) -> np.ndarray:
    """Normalize row labels while keeping tuple/list IDs as one value per row."""

    array = np.asarray(_materialize_iterators(values), dtype=object)
    if array.ndim == 0:
        return _object_value_vector([array.item()])
    if array.ndim == 1:
        return array.reshape(-1)
    if reject_matrix and min(array.shape) > 1:
        raise ValueError(f"{name} must be a one-dimensional vector, not a matrix-shaped array.")
    if 1 in array.shape:
        return array.reshape(-1)
    rows = [tuple(row.tolist()) for row in array.reshape(array.shape[0], -1)]
    return _object_value_vector(rows)


def _is_missing_domain_value(value: Any) -> bool:
    """Return true when a scalar or composite source-domain id is missing."""

    if value is None:
        return True
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _is_missing_domain_value(value.item())
        return any(_is_missing_domain_value(item) for item in value.reshape(-1).tolist())
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_is_missing_domain_value(item) for item in value)
    if isinstance(value, dict):
        return any(_is_missing_domain_value(key) or _is_missing_domain_value(item) for key, item in value.items())

    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    try:
        return bool(np.any(missing))
    except (TypeError, ValueError):
        return False


def _is_missing_domain_array(values: Iterable[Any] | np.ndarray) -> np.ndarray:
    """Return one missing-domain flag per scalar or composite source-domain row."""

    vector = _atomic_vector(values, name="source_domains")
    return np.asarray([_is_missing_domain_value(value) for value in vector], dtype=bool)


def _ordered_unique(values: Iterable[Any] | np.ndarray) -> np.ndarray:
    unique: list[object] = []
    for value in _atomic_vector(values, name="values"):
        if not any(_values_equal(existing, value) for existing in unique):
            unique.append(value)
    return _object_value_vector(unique)


def _encode_atomic(
    values: Iterable[Any] | np.ndarray,
    *,
    name: str,
    reject_matrix: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    vector = _atomic_vector(values, name=name, reject_matrix=reject_matrix)
    unique = _ordered_unique(vector)
    encoded = np.zeros(vector.shape[0], dtype=np.int64)
    for code, value in enumerate(unique):
        encoded[np.asarray([_values_equal(item, value) for item in vector], dtype=bool)] = code
    return unique, encoded


def _stratified_row_fallback_split(
    indices: np.ndarray,
    labels: np.ndarray,
    *,
    validation_fraction: float,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a stratified split that keeps every class on both sides."""

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    _classes, encoded = np.unique(labels, return_inverse=True)
    class_counts = np.bincount(encoded)
    n_classes = int(class_counts.shape[0])
    max_validation_rows = int(np.sum(class_counts - 1))
    desired_validation_rows = int(np.ceil(labels.shape[0] * float(validation_fraction)))
    n_validation_rows = min(max(n_classes, desired_validation_rows), max_validation_rows)

    validation_counts = np.ones(n_classes, dtype=np.int64)
    remaining = int(n_validation_rows - n_classes)
    capacities = class_counts.astype(np.int64, copy=True) - 2
    while remaining > 0 and np.any(capacities > 0):
        candidates = np.flatnonzero(capacities > 0)
        deficits = class_counts[candidates] * float(validation_fraction) - validation_counts[candidates]
        best_deficit = float(np.max(deficits))
        tied = candidates[np.flatnonzero(deficits == best_deficit)]
        chosen = int(tied[np.argmax(capacities[tied])])
        validation_counts[chosen] += 1
        capacities[chosen] -= 1
        remaining -= 1

    rng = np.random.default_rng(random_state)
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    for code in range(n_classes):
        class_indices = indices[encoded == code]
        permuted = rng.permutation(class_indices)
        n_class_validation = int(validation_counts[code])
        validation_parts.append(permuted[:n_class_validation])
        train_parts.append(permuted[n_class_validation:])

    train_idx = rng.permutation(np.concatenate(train_parts)).astype(np.int64, copy=False)
    validation_idx = rng.permutation(np.concatenate(validation_parts)).astype(np.int64, copy=False)
    return train_idx, validation_idx


def install() -> None:
    """Patch source-domain generalization input encoding and row validation splits."""

    _source_ensemble_tuple_domains_patch.install()
    module = importlib.import_module("neureptrace.decoding.source_domain_generalization")
    if getattr(module, _PATCH_MARKER, False):
        return

    module._is_missing_domain_array = _is_missing_domain_array

    def _encode_inputs(source_features: np.ndarray, source_labels: np.ndarray, source_domains: np.ndarray, *, name: str):
        x = np.asarray(source_features, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"{name} source_features must be two-dimensional.")
        if x.shape[0] < 2:
            raise ValueError(f"{name} needs at least two source rows.")

        classes, y = _encode_atomic(source_labels, name="source_labels")
        if y.shape[0] != x.shape[0]:
            raise ValueError("source_features and source_labels must contain the same rows.")
        if classes.shape[0] < 2:
            raise ValueError(f"{name} needs at least two source classes.")

        domain_names, domains = _encode_atomic(source_domains, name="source_domains")
        if domains.shape[0] != x.shape[0]:
            raise ValueError("source_features and source_domains must contain the same rows.")
        if np.any(module._is_missing_domain_array(domain_names)):
            raise ValueError("source_domains must not contain missing values.")
        if domain_names.shape[0] < 2:
            raise ValueError(f"{name} needs at least two source domains/subjects.")
        return x, classes, y.astype(np.int64, copy=False), domain_names, domains.astype(np.int64, copy=False)

    def _source_domain_validation_split(labels: np.ndarray, domains: np.ndarray, *, validation_fraction: float, random_state: int | None):
        labels_array = np.asarray(labels, dtype=np.int64).reshape(-1)
        domains_array = np.asarray(domains, dtype=np.int64).reshape(-1)
        indices = np.arange(labels_array.shape[0])
        unique_domains = np.unique(domains_array)
        n_classes = int(np.unique(labels_array).shape[0])
        fraction = float(validation_fraction)
        rng = np.random.default_rng(random_state)

        if 0.0 < fraction < 1.0 and unique_domains.shape[0] >= 2:
            n_valid_domains = max(1, int(round(unique_domains.shape[0] * fraction)))
            n_valid_domains = min(n_valid_domains, unique_domains.shape[0] - 1)
            if n_valid_domains == 1:
                subsets = [(domain,) for domain in rng.permutation(unique_domains).tolist()]
            else:
                subsets = []
                for _ in range(min(32, 4 * unique_domains.shape[0])):
                    subset = tuple(sorted(rng.choice(unique_domains, size=n_valid_domains, replace=False).tolist()))
                    if subset not in subsets:
                        subsets.append(subset)
            for valid_domains in subsets:
                valid_mask = np.isin(domains_array, valid_domains)
                train_idx = indices[~valid_mask]
                valid_idx = indices[valid_mask]
                if train_idx.size and valid_idx.size:
                    train_has_all_classes = np.unique(labels_array[train_idx]).shape[0] == n_classes
                    valid_has_two_classes = np.unique(labels_array[valid_idx]).shape[0] >= 2
                    if train_has_all_classes and valid_has_two_classes:
                        return train_idx, valid_idx, "heldout_source_domain"

        class_counts = np.bincount(labels_array, minlength=n_classes)
        can_row_validate = 0.0 < fraction < 1.0 and labels_array.shape[0] >= 2 * n_classes and np.min(class_counts) >= 2
        if can_row_validate:
            train_idx, valid_idx = _stratified_row_fallback_split(
                indices,
                labels_array,
                validation_fraction=fraction,
                random_state=random_state,
            )
            return train_idx, valid_idx, "stratified_row_fallback"
        return indices, indices, "training_loss_fallback"

    module._encode_inputs = _encode_inputs
    module._source_domain_validation_split = _source_domain_validation_split
    setattr(module, _PATCH_MARKER, True)


__all__ = ["install"]
