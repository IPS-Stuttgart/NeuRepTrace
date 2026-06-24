"""Weak label-proportion calibration for target-adaptive decoding.

This module implements a deliberately narrow version of learning from label
proportions (LLP) for NeuRepTrace probability traces. It is meant for protocols
where the held-out target block's class proportions are known from task design
or block-level metadata, but no trial-level target labels are used. For the
four-protocol taxonomy, this is a category-2-like weak-supervision protocol and
must be reported separately from ordinary unlabeled target adaptation.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

WEAK_LABEL_PROPORTION_PROTOCOL = "weak_label_proportion_calibration"
WEAK_LABEL_PROPORTION_CATEGORY = "category_2_weak_label_proportion_target_adaptive"


@dataclass(frozen=True, slots=True)
class WeakLabelProportionCalibrationResult:
    """Result of weak target calibration from known class proportions.

    Attributes
    ----------
    probabilities:
        Row-normalized probabilities after label-proportion calibration.
    classes:
        Class order corresponding to the columns in ``probabilities``.
    target_proportions:
        Normalized target proportions in ``classes`` order. This is empty for
        block-wise results because each block can have a different prior.
    class_bias:
        Multiplicative class-bias factors used before row renormalization. This
        is empty for block-wise results.
    iterations:
        Maximum number of iterative-proportional-fitting iterations used. For
        block-wise results this is the maximum over blocks.
    max_mean_proportion_error:
        Maximum absolute discrepancy between requested class proportions and the
        mean calibrated probability distribution. For block-wise results this is
        the maximum discrepancy over blocks.
    converged:
        Whether the requested tolerance was reached in every fitted calibration.
    metadata:
        JSON-serializable protocol/provenance fields.
    block_metadata:
        Per-block provenance rows for block-wise calibration.
    """

    probabilities: np.ndarray
    classes: tuple[Any, ...]
    target_proportions: tuple[float, ...]
    class_bias: tuple[float, ...]
    iterations: int
    max_mean_proportion_error: float
    converged: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    block_metadata: tuple[dict[str, Any], ...] = ()


def normalize_label_proportions(
    target_proportions: Mapping[Any, float] | Sequence[float] | np.ndarray,
    *,
    classes: Sequence[Any] | np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[Any, ...]]:
    """Return normalized class proportions in a stable class order.

    ``target_proportions`` may be either a mapping from class labels to counts or
    proportions, or a sequence already ordered like ``classes``. Missing mapping
    entries are treated as zero only when ``classes`` is supplied explicitly.
    """

    if isinstance(target_proportions, Mapping):
        class_order = tuple(target_proportions.keys()) if classes is None else tuple(classes)
        if not class_order:
            raise ValueError("classes must contain at least one class when target_proportions is a mapping.")
        values = _proportion_values_to_float_array([target_proportions.get(class_label, 0.0) for class_label in class_order])
    else:
        values = _proportion_values_to_float_array(target_proportions)
        class_order = tuple(range(values.size)) if classes is None else tuple(classes)
        if len(class_order) != values.size:
            raise ValueError(
                "target_proportions and classes must have the same length: "
                f"{values.size} != {len(class_order)}."
            )

    if values.ndim != 1 or values.size == 0:
        raise ValueError("target_proportions must contain at least one class proportion.")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("target_proportions must be finite and non-negative.")
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("target_proportions must contain at least one positive entry.")
    return values / total, class_order


def adjust_probabilities_to_label_proportions(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    target_proportions: Mapping[Any, float] | Sequence[float] | np.ndarray,
    *,
    classes: Sequence[Any] | np.ndarray | None = None,
    max_iter: int = 1000,
    tol: float = 1e-9,
    epsilon: float = 1e-12,
) -> WeakLabelProportionCalibrationResult:
    """Calibrate target probabilities so their batch mean matches known proportions.

    The algorithm uses iterative proportional fitting over multiplicative class
    biases:

    ``q[i, c] ∝ p[i, c] * bias[c]``

    and updates ``bias`` until ``mean_i q[i, :]`` matches the supplied target
    class proportions. It uses the target probability matrix and class
    proportions, but never trial-level target labels.
    """

    matrix = _as_probability_matrix(probabilities, epsilon=epsilon)
    target_prior, class_order = normalize_label_proportions(target_proportions, classes=classes)
    if matrix.shape[1] != target_prior.size:
        raise ValueError(
            "probabilities and target_proportions must contain the same number of classes: "
            f"{matrix.shape[1]} != {target_prior.size}."
        )

    max_iter = _normalize_positive_int(max_iter, name="max_iter")
    tol = _normalize_nonnegative_float(tol, name="tol")
    epsilon = _normalize_positive_float(epsilon, name="epsilon")

    class_bias = np.ones(matrix.shape[1], dtype=float)
    class_bias[target_prior == 0.0] = 0.0
    active = target_prior > 0.0
    adjusted = _apply_class_bias(matrix, class_bias, epsilon=epsilon)
    error = float(np.max(np.abs(np.mean(adjusted, axis=0) - target_prior)))
    iterations = 0

    for iterations in range(1, max_iter + 1):
        adjusted = _apply_class_bias(matrix, class_bias, epsilon=epsilon)
        mean_probability = np.mean(adjusted, axis=0)
        error = float(np.max(np.abs(mean_probability - target_prior)))
        if error <= tol:
            break
        update = np.ones_like(class_bias)
        update[active] = target_prior[active] / np.maximum(mean_probability[active], epsilon)
        update[~active] = 0.0
        class_bias *= update
        scale = float(np.max(class_bias))
        if scale > 0.0:
            class_bias /= scale

    adjusted = _apply_class_bias(matrix, class_bias, epsilon=epsilon)
    mean_probability = np.mean(adjusted, axis=0)
    error = float(np.max(np.abs(mean_probability - target_prior)))
    converged = bool(error <= tol)
    metadata = _base_metadata(
        n_samples=matrix.shape[0],
        n_classes=matrix.shape[1],
        iterations=iterations,
        max_mean_proportion_error=error,
        converged=converged,
        blockwise=False,
    )
    metadata.update(
        {
            "target_proportions": "|".join(f"{value:.12g}" for value in target_prior),
            "mean_calibrated_probabilities": "|".join(f"{value:.12g}" for value in mean_probability),
        }
    )
    return WeakLabelProportionCalibrationResult(
        probabilities=adjusted,
        classes=class_order,
        target_proportions=tuple(float(value) for value in target_prior),
        class_bias=tuple(float(value) for value in class_bias),
        iterations=int(iterations),
        max_mean_proportion_error=error,
        converged=converged,
        metadata=metadata,
    )


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
) -> WeakLabelProportionCalibrationResult:
    """Apply weak label-proportion calibration independently per target block.

    This is useful for oddball or ERP designs where each target run/block has a
    known target/non-target ratio but individual trial labels are hidden from the
    adaptation algorithm.
    """

    matrix = _as_probability_matrix(probabilities, epsilon=epsilon)
    block_vector = np.asarray(block_ids, dtype=object).reshape(-1)
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
    ordered_blocks = tuple(dict.fromkeys(block_vector.tolist()))
    if not ordered_blocks:
        raise ValueError("At least one block is required for block-wise label-proportion calibration.")

    for block in ordered_blocks:
        mask = block_vector == block
        proportions = _lookup_block_proportions(target_proportions_by_block, block, default_proportions=default_proportions)
        result = adjust_probabilities_to_label_proportions(
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

    metadata = _base_metadata(
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
    return WeakLabelProportionCalibrationResult(
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


def predict_labels_from_label_proportions(result: WeakLabelProportionCalibrationResult) -> np.ndarray:
    """Return argmax labels from a label-proportion calibrated result."""

    class_array = np.asarray(result.classes, dtype=object)
    return class_array[np.argmax(result.probabilities, axis=1)]


def _as_probability_matrix(probabilities: Sequence[Sequence[float]] | np.ndarray, *, epsilon: float) -> np.ndarray:
    epsilon = _normalize_positive_float(epsilon, name="epsilon")
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional matrix.")
    if matrix.shape[0] < 1 or matrix.shape[1] < 2:
        raise ValueError("probabilities must contain at least one row and two classes.")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("probabilities must be finite and non-negative.")
    row_sums = np.sum(matrix, axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Each probability row must contain at least one positive value.")
    matrix = matrix / row_sums
    matrix = np.clip(matrix, epsilon, None)
    return matrix / np.sum(matrix, axis=1, keepdims=True)


def _apply_class_bias(probabilities: np.ndarray, class_bias: np.ndarray, *, epsilon: float) -> np.ndarray:
    weighted = probabilities * class_bias.reshape(1, -1)
    row_sums = np.sum(weighted, axis=1, keepdims=True)
    if np.any(row_sums <= epsilon):
        raise ValueError("Label-proportion calibration produced a zero-probability row; check proportions and input probabilities.")
    return weighted / row_sums


def _lookup_block_proportions(
    target_proportions_by_block: Mapping[Hashable, Mapping[Any, float] | Sequence[float] | np.ndarray],
    block: Hashable,
    *,
    default_proportions: Mapping[Any, float] | Sequence[float] | np.ndarray | None,
) -> Mapping[Any, float] | Sequence[float] | np.ndarray:
    try:
        return target_proportions_by_block[block]
    except KeyError:
        text_block = str(block)
        if text_block in target_proportions_by_block:
            return target_proportions_by_block[text_block]
        if default_proportions is not None:
            return default_proportions
        raise KeyError(f"Missing target label proportions for block {block!r}.") from None


def _proportion_values_to_float_array(values: Any) -> np.ndarray:
    raw = np.asarray(values, dtype=object).reshape(-1)
    if any(isinstance(value, (bool, np.bool_)) for value in raw):
        raise ValueError("target_proportions must be numeric counts or proportions, not boolean flags.")
    try:
        return raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("target_proportions must be finite and non-negative.") from exc


def _base_metadata(
    *,
    n_samples: int,
    n_classes: int,
    iterations: int,
    max_mean_proportion_error: float,
    converged: bool,
    blockwise: bool,
) -> dict[str, Any]:
    return {
        "protocol": WEAK_LABEL_PROPORTION_PROTOCOL,
        "protocol_category": WEAK_LABEL_PROPORTION_CATEGORY,
        "protocol_note": "uses known target class proportions or block-level task constraints, but no trial-level target labels",
        "uses_unlabeled_target_data": True,
        "uses_target_trial_labels": False,
        "uses_target_label_proportions": True,
        "uses_weak_task_constraints": True,
        "valid_for_strict_source_only": False,
        "valid_for_ordinary_unlabeled_target_adaptation": False,
        "n_samples": int(n_samples),
        "n_classes": int(n_classes),
        "blockwise": bool(blockwise),
        "iterations": int(iterations),
        "max_mean_proportion_error": float(max_mean_proportion_error),
        "converged": bool(converged),
    }


def _normalize_positive_int(value: int | str, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    parsed = int(value)
    if parsed < 1 or float(parsed) != float(value):
        raise ValueError(f"{name} must be a positive integer.")
    return parsed


def _normalize_positive_float(value: float | str, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be positive and finite.")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed


def _normalize_nonnegative_float(value: float | str, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be non-negative and finite.")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be non-negative and finite.")
    return parsed
