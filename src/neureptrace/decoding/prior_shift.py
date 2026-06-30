"""Unlabeled target prior-shift adaptation for probability traces."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence, Set
from dataclasses import dataclass, field
from typing import Any

import numpy as np

PRIOR_SHIFT_PROTOCOL = "unlabeled_target_prior_shift_adaptation"
PRIOR_SHIFT_CATEGORY = "2_unlabeled_target_adaptive"
EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class PriorShiftAdaptationResult:
    """Probability rows after unlabeled prior-shift adaptation."""

    probabilities: np.ndarray
    target_prior: np.ndarray
    source_prior: np.ndarray
    class_bias: np.ndarray
    n_iterations: int
    converged: bool
    max_delta: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PriorShiftBlockResult:
    """Block-wise prior-shift adaptation output."""

    probabilities: np.ndarray
    block_results: Mapping[Hashable, PriorShiftAdaptationResult]
    metadata: dict[str, Any] = field(default_factory=dict)


def adapt_probabilities_for_prior_shift(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    source_prior: Sequence[float] | np.ndarray | None = None,
    initial_target_prior: Sequence[float] | np.ndarray | None = None,
    target_prior: Sequence[float] | np.ndarray | None = None,
    max_iter: int | str = 100,
    tol: float | str = 1e-8,
    smoothing: float | str = 1e-6,
    damping: float | str = 1.0,
    epsilon: float | str = EPSILON,
) -> PriorShiftAdaptationResult:
    """Estimate a target class prior from unlabeled probability rows and reweight.

    The function accepts source-model posteriors for target rows and optionally a
    source training prior. It never accepts target class labels. If ``target_prior``
    is supplied, EM is skipped and only prior-ratio reweighting is applied.
    """

    matrix = _probability_matrix(probabilities, epsilon=epsilon)
    n_rows, n_classes = matrix.shape
    eps = _positive_float(epsilon, name="epsilon")
    source = _prior(source_prior, n_classes=n_classes, default="uniform", name="source_prior", epsilon=eps)
    smooth = _nonnegative_float(smoothing, name="smoothing")
    max_iterations = _positive_int(max_iter, name="max_iter")
    tolerance = _nonnegative_float(tol, name="tol")
    damp = _unit_interval_float(damping, name="damping")

    if target_prior is not None:
        target = _prior(target_prior, n_classes=n_classes, default="uniform", name="target_prior", epsilon=eps)
        iterations = 0
        converged = True
        max_delta = 0.0
        mode = "fixed_target_prior"
    else:
        if initial_target_prior is None:
            target = _normalize_prior(np.mean(matrix, axis=0), epsilon=eps)
        else:
            target = _prior(initial_target_prior, n_classes=n_classes, default="uniform", name="initial_target_prior", epsilon=eps)
        target = _smooth(target, smooth, eps)
        iterations = 0
        converged = False
        max_delta = float("inf")
        mode = "em_estimated_target_prior"
        for iterations in range(1, max_iterations + 1):
            responsibilities = reweight_probabilities_by_prior(matrix, source_prior=source, target_prior=target, epsilon=eps)
            updated = _smooth(np.mean(responsibilities, axis=0), smooth, eps)
            if damp < 1.0:
                updated = _normalize_prior((1.0 - damp) * target + damp * updated, epsilon=eps)
            max_delta = float(np.max(np.abs(updated - target)))
            target = updated
            if max_delta <= tolerance:
                converged = True
                break

    adapted = reweight_probabilities_by_prior(matrix, source_prior=source, target_prior=target, epsilon=eps)
    class_bias = _normalize_bias(target / np.maximum(source, eps))
    metadata = _metadata(
        mode=mode,
        n_rows=n_rows,
        n_classes=n_classes,
        source_prior=source,
        target_prior=target,
        class_bias=class_bias,
        iterations=iterations,
        converged=converged,
        max_delta=max_delta,
        blockwise=False,
    )
    return PriorShiftAdaptationResult(adapted, target, source, class_bias, int(iterations), bool(converged), float(max_delta), metadata)


def reweight_probabilities_by_prior(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    source_prior: Sequence[float] | np.ndarray,
    target_prior: Sequence[float] | np.ndarray,
    epsilon: float | str = EPSILON,
) -> np.ndarray:
    """Reweight posterior rows by ``target_prior / source_prior``."""

    matrix = _probability_matrix(probabilities, epsilon=epsilon)
    eps = _positive_float(epsilon, name="epsilon")
    source = _prior(source_prior, n_classes=matrix.shape[1], default="uniform", name="source_prior", epsilon=eps)
    target = _prior(target_prior, n_classes=matrix.shape[1], default="uniform", name="target_prior", epsilon=eps)
    return _normalize_rows(matrix * (target / np.maximum(source, eps))[None, :], epsilon=eps)


def adapt_probability_blocks_for_prior_shift(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    block_ids: Sequence[Hashable] | np.ndarray,
    *,
    source_prior: Sequence[float] | np.ndarray | None = None,
    min_block_rows: int | str = 2,
    **kwargs: Any,
) -> PriorShiftBlockResult:
    """Run prior-shift adaptation separately for each target block."""

    matrix = _probability_matrix(probabilities, epsilon=kwargs.get("epsilon", EPSILON))
    blocks = _object_vector(block_ids, expected_length=matrix.shape[0], name="block_ids")
    minimum = _positive_int(min_block_rows, name="min_block_rows")
    adapted = np.empty_like(matrix)
    results: dict[Hashable, PriorShiftAdaptationResult] = {}
    for block in _unique_values(blocks):
        mask = _object_equal_mask(blocks, block)
        if int(np.sum(mask)) < minimum:
            raise ValueError(f"Block {block!r} has fewer than min_block_rows={minimum} rows.")
        result = adapt_probabilities_for_prior_shift(matrix[mask], source_prior=source_prior, **kwargs)
        adapted[mask] = result.probabilities
        results[block] = result
    metadata = {
        "prior_shift_adaptation": True,
        "prior_shift_protocol": PRIOR_SHIFT_PROTOCOL,
        "prior_shift_protocol_category": PRIOR_SHIFT_CATEGORY,
        "prior_shift_uses_target_probabilities": True,
        "prior_shift_uses_target_labels": False,
        "prior_shift_blockwise": True,
        "prior_shift_n_rows": int(matrix.shape[0]),
        "prior_shift_n_classes": int(matrix.shape[1]),
        "prior_shift_n_blocks": int(len(results)),
        "prior_shift_blocks": "|".join(str(block) for block in results),
        "prior_shift_converged_all_blocks": all(result.converged for result in results.values()),
        "prior_shift_target_priors_by_block": "|".join(f"{block}:{_format(result.target_prior)}" for block, result in results.items()),
    }
    return PriorShiftBlockResult(adapted, results, metadata)


def prior_from_labels(labels: Sequence[Hashable] | np.ndarray, classes: Sequence[Hashable] | np.ndarray | None = None, *, smoothing: float | str = 0.0) -> tuple[np.ndarray, tuple[Hashable, ...]]:
    """Compute an empirical prior from source labels."""

    label_vector = _object_vector(labels, name="labels")
    if label_vector.size == 0:
        raise ValueError("labels must contain at least one row.")
    class_order = _unique_values(label_vector) if classes is None else tuple(_object_vector(classes, name="classes").tolist())
    counts = np.asarray([np.count_nonzero(_object_equal_mask(label_vector, class_label)) for class_label in class_order], dtype=float)
    return _normalize_prior(counts + _nonnegative_float(smoothing, name="smoothing"), epsilon=EPSILON), class_order


def _metadata(*, mode: str, n_rows: int, n_classes: int, source_prior: np.ndarray, target_prior: np.ndarray, class_bias: np.ndarray, iterations: int, converged: bool, max_delta: float, blockwise: bool) -> dict[str, Any]:
    return {
        "prior_shift_adaptation": True,
        "prior_shift_protocol": PRIOR_SHIFT_PROTOCOL,
        "prior_shift_protocol_category": PRIOR_SHIFT_CATEGORY,
        "prior_shift_mode": mode,
        "prior_shift_uses_target_probabilities": True,
        "prior_shift_uses_target_labels": False,
        "prior_shift_valid_for_strict_source_only": False,
        "prior_shift_valid_for_unlabeled_target_adaptation": True,
        "prior_shift_valid_for_benchmark": False,
        "prior_shift_blockwise": bool(blockwise),
        "prior_shift_n_rows": int(n_rows),
        "prior_shift_n_classes": int(n_classes),
        "prior_shift_n_iterations": int(iterations),
        "prior_shift_converged": bool(converged),
        "prior_shift_max_delta": float(max_delta),
        "prior_shift_source_prior": _format(source_prior),
        "prior_shift_target_prior": _format(target_prior),
        "prior_shift_class_bias": _format(class_bias),
    }


def _probability_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, epsilon: float | str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 2:
        raise ValueError("probabilities must be a non-empty two-dimensional matrix with at least two classes.")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("probabilities must contain finite non-negative values.")
    return _normalize_rows(matrix, epsilon=_positive_float(epsilon, name="epsilon"))


def _normalize_rows(matrix: np.ndarray, *, epsilon: float) -> np.ndarray:
    row_sums = np.sum(np.maximum(matrix, 0.0), axis=1, keepdims=True)
    if np.any(row_sums <= epsilon):
        raise ValueError("Each probability row must have positive mass.")
    return np.maximum(matrix, 0.0) / row_sums


def _prior(values: Sequence[float] | np.ndarray | None, *, n_classes: int, default: str, name: str, epsilon: float) -> np.ndarray:
    if values is None:
        vector = np.full(n_classes, 1.0 / n_classes, dtype=float) if default == "uniform" else np.ones(n_classes, dtype=float)
    else:
        vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.shape[0] != n_classes:
        raise ValueError(f"{name} must contain one value per class: {vector.shape[0]} != {n_classes}.")
    if not np.all(np.isfinite(vector)) or np.any(vector < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values.")
    return _normalize_prior(vector, epsilon=epsilon)


def _normalize_prior(values: np.ndarray, *, epsilon: float) -> np.ndarray:
    vector = np.maximum(np.asarray(values, dtype=float).reshape(-1), 0.0)
    total = float(np.sum(vector))
    if total <= epsilon:
        raise ValueError("Prior vector must have positive mass.")
    return vector / total


def _smooth(values: np.ndarray, smoothing: float, epsilon: float) -> np.ndarray:
    return _normalize_prior(np.maximum(np.asarray(values, dtype=float).reshape(-1) + smoothing, epsilon), epsilon=epsilon)


def _normalize_bias(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    mean = float(np.mean(vector))
    return np.ones_like(vector) if mean <= 0.0 or not np.isfinite(mean) else vector / mean


def _format(values: np.ndarray) -> str:
    return "|".join(f"{float(value):.12g}" for value in np.asarray(values, dtype=float).reshape(-1))


def _object_vector(values: Sequence[Any] | np.ndarray, *, name: str, expected_length: int | None = None) -> np.ndarray:
    items = _row_items(values, expected_length=expected_length)
    if expected_length is not None and len(items) != expected_length:
        raise ValueError(f"{name} must contain one value per probability row: {len(items)} != {expected_length}.")
    vector = np.empty(len(items), dtype=object)
    for index, value in enumerate(items):
        vector[index] = _hashable_object_value(value)
    return vector


def _row_items(values: Sequence[Any] | np.ndarray, *, expected_length: int | None) -> list[Any]:
    if isinstance(values, np.ndarray):
        array = np.asarray(values, dtype=object)
        if array.ndim == 0:
            return [array.item()]
        if array.ndim == 1:
            return array.tolist()
        rows = array.reshape(array.shape[0], -1)
        if expected_length is not None and rows.shape[0] == expected_length:
            return [row[0] if row.shape[0] == 1 else tuple(row.tolist()) for row in rows]
        if expected_length is not None and array.size == expected_length and 1 in array.shape:
            return array.reshape(-1).tolist()
        return [tuple(row.tolist()) for row in rows]
    if isinstance(values, (str, bytes)):
        return [values]
    try:
        return list(values)
    except TypeError:
        return [values]


def _hashable_object_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _hashable_object_value(value.item())
        return tuple(_hashable_object_value(item) for item in value.tolist())
    if isinstance(value, Mapping):
        pairs = ((_hashable_object_value(key), _hashable_object_value(item)) for key, item in value.items())
        return tuple(sorted(pairs, key=lambda pair: repr(pair[0])))
    if isinstance(value, Set) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(sorted((_hashable_object_value(item) for item in value), key=repr))
    if isinstance(value, list):
        return tuple(_hashable_object_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_hashable_object_value(item) for item in value)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _unique_values(values: Sequence[Any] | np.ndarray) -> tuple[Any, ...]:
    unique: list[Any] = []
    vector = values if isinstance(values, np.ndarray) else _object_vector(values, name="values")
    for value in vector.tolist():
        if not any(_object_equal(value, existing) for existing in unique):
            unique.append(value)
    return tuple(unique)


def _object_equal_mask(values: np.ndarray, target: Any) -> np.ndarray:
    return np.asarray([_object_equal(value, target) for value in values.tolist()], dtype=bool)


def _object_equal(left: Any, right: Any) -> bool:
    try:
        comparison = left == right
    except (TypeError, ValueError):
        return False
    if isinstance(comparison, (bool, np.bool_)):
        return bool(comparison)
    try:
        return bool(np.all(comparison))
    except (TypeError, ValueError):
        return False


def _numeric_scalar(value: object, *, message: str) -> float:
    if isinstance(value, (bool, np.bool_)) or isinstance(value, np.ndarray):
        raise ValueError(message)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc


def _positive_int(value: int | str, *, name: str) -> int:
    message = f"{name} must be a positive integer."
    number = _numeric_scalar(value, message=message)
    if not np.isfinite(number) or number % 1.0 != 0.0 or number < 1:
        raise ValueError(message)
    return int(number)


def _positive_float(value: float | str, *, name: str) -> float:
    message = f"{name} must be positive and finite."
    number = _numeric_scalar(value, message=message)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(message)
    return number


def _nonnegative_float(value: float | str, *, name: str) -> float:
    message = f"{name} must be finite and non-negative."
    number = _numeric_scalar(value, message=message)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(message)
    return number


def _unit_interval_float(value: float | str, *, name: str) -> float:
    number = _positive_float(value, name=name)
    if number > 1.0:
        raise ValueError(f"{name} must be in (0, 1].")
    return number
