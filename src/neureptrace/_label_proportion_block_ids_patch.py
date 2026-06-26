"""Preserve composite block ids in weak label-proportion calibration."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np

import neureptrace.decoding.label_proportions as _label_proportions

_PATCH_MARKER = "_neureptrace_label_proportion_block_ids_patch_installed"


def _object_block_vector(values: Sequence[Hashable] | np.ndarray) -> np.ndarray:
    """Return one hashable block id per probability row without flattening tuples."""

    if isinstance(values, np.ndarray):
        array = np.asarray(values, dtype=object)
        if array.ndim == 0:
            items = [array.item()]
        elif array.ndim == 1:
            items = array.tolist()
        elif array.ndim == 2 and 1 in array.shape:
            items = array.reshape(-1).tolist()
        elif array.ndim == 2:
            items = [tuple(row.tolist()) for row in array]
        else:
            raise ValueError(f"block_ids must be one-dimensional, a single-column vector, or a two-dimensional composite-id matrix; got shape {array.shape}.")
    elif isinstance(values, (str, bytes)):
        items = [values]
    else:
        try:
            items = list(values)
        except TypeError:
            items = [values]

    vector = np.empty(len(items), dtype=object)
    for index, item in enumerate(items):
        try:
            hash(item)
        except TypeError as exc:
            raise ValueError(f"block_ids must contain hashable block identifiers; got {item!r}.") from exc
        vector[index] = item
    return vector


def _values_equal(left: Any, right: Any) -> bool:
    try:
        result = left == right
    except Exception:  # pragma: no cover - defensive fallback for unusual metadata objects
        return False
    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    try:
        return bool(np.all(result))
    except Exception:  # pragma: no cover - defensive fallback for unusual metadata objects
        return False


def _block_equal_mask(block_vector: np.ndarray, block: Hashable) -> np.ndarray:
    return np.asarray([_values_equal(value, block) for value in block_vector.tolist()], dtype=bool)


def _unique_blocks(block_vector: np.ndarray) -> tuple[Hashable, ...]:
    blocks: list[Hashable] = []
    for block in block_vector.tolist():
        if not any(_values_equal(block, existing) for existing in blocks):
            blocks.append(block)
    return tuple(blocks)


def _adjust_probability_blocks_to_label_proportions(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    block_ids: Sequence[Hashable] | np.ndarray,
    target_proportions_by_block: Mapping[Hashable, Mapping[Any, float] | Sequence[float] | np.ndarray],
    *,
    classes: Sequence[Any] | np.ndarray | None = None,
    default_proportions: Mapping[Any, float] | Sequence[float] | np.ndarray | None = None,
    max_iter: int = 1000,
    tol: float = 1e-9,
    epsilon: float = 1e-12,
) -> _label_proportions.WeakLabelProportionCalibrationResult:
    """Apply block-wise label-proportion calibration with atomic block ids."""

    matrix = _label_proportions._as_probability_matrix(probabilities, epsilon=epsilon)
    block_vector = _object_block_vector(block_ids)
    if block_vector.shape[0] != matrix.shape[0]:
        raise ValueError("block_ids must have the same row count as probabilities.")
    if not isinstance(target_proportions_by_block, Mapping):
        raise ValueError("target_proportions_by_block must be a mapping from block id to proportions.")

    adjusted = np.empty_like(matrix)
    block_rows: list[dict[str, Any]] = []
    block_classes: tuple[Any, ...] | None = None
    max_error = 0.0
    max_iterations = 0
    all_converged = True
    ordered_blocks = _unique_blocks(block_vector)
    if not ordered_blocks:
        raise ValueError("At least one block is required for block-wise label-proportion calibration.")

    for block in ordered_blocks:
        mask = _block_equal_mask(block_vector, block)
        proportions = _label_proportions._lookup_block_proportions(
            target_proportions_by_block,
            block,
            default_proportions=default_proportions,
        )
        result = _label_proportions.adjust_probabilities_to_label_proportions(
            matrix[mask],
            proportions,
            classes=classes,
            max_iter=max_iter,
            tol=tol,
            epsilon=epsilon,
        )
        if block_classes is None:
            block_classes = result.classes
        elif block_classes != result.classes:
            raise ValueError("All blocks must use the same class order.")
        adjusted[mask] = result.probabilities
        max_error = max(max_error, result.max_mean_proportion_error)
        max_iterations = max(max_iterations, result.iterations)
        all_converged = bool(all_converged and result.converged)
        block_rows.append(
            {
                "block": str(block),
                "n_samples": int(np.sum(mask)),
                "iterations": int(result.iterations),
                "max_mean_proportion_error": float(result.max_mean_proportion_error),
                "converged": bool(result.converged),
                "target_proportions": "|".join(f"{value:.12g}" for value in result.target_proportions),
            }
        )

    metadata = _label_proportions._base_metadata(
        n_samples=matrix.shape[0],
        n_classes=matrix.shape[1],
        iterations=max_iterations,
        max_mean_proportion_error=max_error,
        converged=all_converged,
        blockwise=True,
    )
    metadata.update(
        {
            "n_blocks": len(ordered_blocks),
            "min_block_samples": int(min(row["n_samples"] for row in block_rows)),
            "max_block_samples": int(max(row["n_samples"] for row in block_rows)),
        }
    )
    return _label_proportions.WeakLabelProportionCalibrationResult(
        probabilities=adjusted,
        classes=tuple(range(matrix.shape[1])) if block_classes is None else block_classes,
        target_proportions=(),
        class_bias=(),
        iterations=max_iterations,
        max_mean_proportion_error=max_error,
        converged=all_converged,
        metadata=metadata,
        block_metadata=tuple(block_rows),
    )


def install() -> None:
    """Install tuple-safe block-id handling for weak label-proportion calibration."""

    current = _label_proportions.adjust_probability_blocks_to_label_proportions
    if getattr(current, _PATCH_MARKER, False):
        return
    setattr(_adjust_probability_blocks_to_label_proportions, _PATCH_MARKER, True)
    _label_proportions.adjust_probability_blocks_to_label_proportions = _adjust_probability_blocks_to_label_proportions


__all__ = ["install"]
