"""Validate block-id vectors for weak label-proportion calibration."""

from __future__ import annotations

import importlib
from collections.abc import Hashable, Mapping, Sequence
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_label_proportion_block_ids_patch_installed"


def _as_object_vector(values: Sequence[Any]) -> np.ndarray:
    vector = np.empty(len(values), dtype=object)
    vector[:] = list(values)
    return vector


def _block_vector_from_array(block_ids: np.ndarray) -> np.ndarray:
    array = np.asarray(block_ids, dtype=object)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim == 1:
        return array.astype(object, copy=False)
    if array.ndim == 2 and 1 in array.shape:
        return array.reshape(-1).astype(object, copy=False)
    raise ValueError("block_ids must be a one-dimensional vector of block identifiers, not a matrix.")


def _block_vector_from_input(block_ids: Any) -> np.ndarray:
    if isinstance(block_ids, np.ndarray):
        return _block_vector_from_array(block_ids)
    if isinstance(block_ids, (str, bytes)):
        return _as_object_vector([block_ids])
    try:
        values = list(block_ids)
    except TypeError:
        return _as_object_vector([block_ids])
    return _as_object_vector(values)


def _validate_hashable_block_ids(block_vector: np.ndarray) -> None:
    for block in block_vector.tolist():
        if not isinstance(block, Hashable):
            raise ValueError("block_ids must be a one-dimensional sequence of hashable block identifiers.")
        try:
            hash(block)
        except TypeError as exc:
            raise ValueError("block_ids must be a one-dimensional sequence of hashable block identifiers.") from exc


def _normalize_block_ids(block_ids: Any, *, n_rows: int) -> np.ndarray:
    block_vector = _block_vector_from_input(block_ids)
    if block_vector.ndim != 1:
        raise ValueError("block_ids must be a one-dimensional vector of block identifiers, not a matrix.")
    if block_vector.shape[0] != n_rows:
        raise ValueError("block_ids must have the same row count as probabilities.")
    _validate_hashable_block_ids(block_vector)
    return block_vector


def install() -> None:
    """Patch block-wise label-proportion calibration block-id parsing."""

    label_proportions = importlib.import_module("neureptrace.decoding.label_proportions")
    if getattr(label_proportions, _PATCH_MARKER, False):
        return

    original_adjust_blocks = label_proportions.adjust_probability_blocks_to_label_proportions

    @wraps(original_adjust_blocks)
    def adjust_probability_blocks_to_label_proportions(
        probabilities: Sequence[Sequence[float]] | np.ndarray,
        block_ids: Sequence[Hashable] | np.ndarray,
        target_proportions_by_block: Mapping[Hashable, Mapping[Any, float] | Sequence[float] | np.ndarray],
        *,
        classes: Sequence[Any] | np.ndarray | None = None,
        default_proportions: Mapping[Any, float] | Sequence[float] | np.ndarray | None = None,
        max_iter: int = 1000,
        tol: float = 1e-9,
        epsilon: float = 1e-12,
    ) -> Any:
        matrix = label_proportions._as_probability_matrix(probabilities, epsilon=epsilon)
        block_vector = _normalize_block_ids(block_ids, n_rows=matrix.shape[0])
        return original_adjust_blocks(
            matrix,
            block_vector,
            target_proportions_by_block,
            classes=classes,
            default_proportions=default_proportions,
            max_iter=max_iter,
            tol=tol,
            epsilon=epsilon,
        )

    label_proportions.adjust_probability_blocks_to_label_proportions = adjust_probability_blocks_to_label_proportions
    setattr(label_proportions, _PATCH_MARKER, True)


__all__ = ["install"]
