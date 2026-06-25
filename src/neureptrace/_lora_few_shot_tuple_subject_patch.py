"""Treat composite source-subject identifiers atomically in LoRA episodes."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_lora_few_shot_tuple_subject_patch_installed"


def _object_value_vector(values: Sequence[Any]) -> np.ndarray:
    vector = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        vector[index] = value
    return vector


def _atomic_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=object)
        if array.ndim == 0:
            return array.item()
        return tuple(array.reshape(-1).tolist())
    if isinstance(value, list):
        return tuple(value)
    return value


def _object_vector(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    if array.ndim == 0:
        return _object_value_vector([array.item()])
    if array.ndim == 1:
        return _object_value_vector([_atomic_value(value) for value in array.reshape(-1).tolist()])
    rows = [tuple(row.tolist()) for row in array.reshape(array.shape[0], -1)]
    return _object_value_vector(rows)


def _values_equal(left: Any, right: Any) -> bool:
    left = _atomic_value(left)
    right = _atomic_value(right)
    try:
        equal = left == right
    except (TypeError, ValueError):
        return False
    try:
        return bool(equal)
    except (TypeError, ValueError):
        return False


def _value_mask(values: Sequence[Any] | np.ndarray, target: Any) -> np.ndarray:
    return np.asarray([_values_equal(value, target) for value in _object_vector(values)], dtype=bool)


def install() -> None:
    """Patch LoRA source-subject episode selection for tuple identifiers."""

    lora_few_shot = importlib.import_module("neureptrace.decoding.lora_few_shot")
    if getattr(lora_few_shot, _PATCH_MARKER, False):
        return

    original_balanced_subject_episode_indices = lora_few_shot._balanced_subject_episode_indices

    @wraps(original_balanced_subject_episode_indices)
    def _balanced_subject_episode_indices(
        y: np.ndarray,
        subjects: Sequence[Any] | np.ndarray,
        subject: Any,
        *,
        support_per_class: int,
        query_per_class: int,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        y = np.asarray(y)
        subject_mask = _value_mask(subjects, subject)
        if subject_mask.shape[0] != y.shape[0]:
            raise ValueError("subjects must contain one value per encoded source-label row.")

        rng = np.random.default_rng(seed)
        support_count = int(support_per_class)
        query_count = int(query_per_class)
        support_parts: list[np.ndarray] = []
        query_parts: list[np.ndarray] = []
        for class_label in np.unique(y):
            positions = np.flatnonzero(subject_mask & (y == class_label))
            required = support_count + query_count
            if positions.size < required:
                return np.array([], dtype=int), np.array([], dtype=int)
            shuffled = rng.permutation(positions)
            support_parts.append(shuffled[:support_count])
            query_parts.append(shuffled[support_count:required])
        return (
            np.sort(np.concatenate(support_parts).astype(int, copy=False)),
            np.sort(np.concatenate(query_parts).astype(int, copy=False)),
        )

    lora_few_shot._balanced_subject_episode_indices = _balanced_subject_episode_indices
    setattr(lora_few_shot, _PATCH_MARKER, True)


__all__ = ["install"]
