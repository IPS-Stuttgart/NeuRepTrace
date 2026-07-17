"""Source-only range selector."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_RANGE_SELECTOR_PROTOCOL = "strict_source_only_range_selector"
SOURCE_RANGE_SELECTOR_CATEGORY = "1_strict_source_only"


@dataclass(frozen=True, slots=True)
class SourceRangeSelectorResult:
    train_features: np.ndarray
    test_features: np.ndarray
    selected_indices: np.ndarray
    ranges: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def fit_source_range_selector(*, source_features, test_features, min_range: float = 0.0, top_k: int | None = None) -> SourceRangeSelectorResult:
    source = _matrix(source_features, name="source_features")
    test = _matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError("feature widths differ")
    min_range_value = _nonnegative_float(min_range, name="min_range")
    top_k_value = None if top_k is None else _positive_int(top_k, name="top_k")
    ranges = _stable_column_ranges(source)
    selected = select_source_range_features(ranges, min_range=min_range_value, top_k=top_k_value)
    return SourceRangeSelectorResult(
        train_features=_compact_float32(source[:, selected]),
        test_features=_compact_float32(test[:, selected]),
        selected_indices=selected.astype(int, copy=False),
        ranges=_compact_float32(ranges),
        metadata={
            "source_range_selector": True,
            "source_range_selector_protocol": SOURCE_RANGE_SELECTOR_PROTOCOL,
            "source_range_selector_protocol_category": SOURCE_RANGE_SELECTOR_CATEGORY,
            "source_range_selector_uses_source_features": True,
            "source_range_selector_uses_test_features_for_fitting": False,
            "source_range_selector_valid_for_strict_source_only": True,
            "source_range_selector_input_dim": int(source.shape[1]),
            "source_range_selector_output_dim": int(selected.shape[0]),
            "source_range_selector_min_range": min_range_value,
            "source_range_selector_top_k": "" if top_k_value is None else top_k_value,
        },
    )


def select_source_range_features(ranges, *, min_range: float = 0.0, top_k: int | None = None) -> np.ndarray:
    materialized = _materialize_one_pass_iterables(ranges)
    if _contains_boolean_value(materialized):
        raise ValueError("ranges must contain finite non-negative numeric, non-boolean values.")
    values = np.asarray(materialized, dtype=float)
    if values.ndim != 1:
        raise ValueError("ranges must be one-dimensional.")
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("bad ranges")
    threshold = _nonnegative_float(min_range, name="min_range")
    selected = np.flatnonzero(values > threshold)
    if top_k is not None:
        k = _positive_int(top_k, name="top_k")
        ranked = np.argsort(values, kind="mergesort")[-min(k, values.size):]
        selected = np.intersect1d(selected, ranked, assume_unique=False)
    if selected.size == 0:
        selected = np.asarray([int(np.argmax(values))], dtype=int)
    return np.sort(selected).astype(int, copy=False)


def _stable_column_ranges(matrix: np.ndarray) -> np.ndarray:
    """Compute finite column ranges without overflowing max-minus-min."""

    minimum = np.min(matrix, axis=0)
    maximum = np.max(matrix, axis=0)
    magnitude = np.maximum(np.abs(minimum), np.abs(maximum))
    normalized_range = np.zeros_like(magnitude)
    nonzero = magnitude > 0.0
    normalized_range[nonzero] = maximum[nonzero] / magnitude[nonzero] - minimum[nonzero] / magnitude[nonzero]
    normalized_range = np.maximum(normalized_range, 0.0)

    ranges = np.zeros_like(magnitude)
    positive = normalized_range > 0.0
    if np.any(positive):
        normalized_positive = normalized_range[positive]
        magnitude_positive = magnitude[positive]
        maximum_magnitude = np.finfo(float).max / normalized_positive
        safe = magnitude_positive <= maximum_magnitude
        scaled = np.empty_like(normalized_positive)
        scaled[safe] = normalized_positive[safe] * magnitude_positive[safe]
        scaled[~safe] = np.finfo(float).max
        ranges[positive] = scaled
    return ranges


def _compact_float32(values: np.ndarray) -> np.ndarray:
    """Use float32 only when conversion preserves finite, nonzero values."""

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        compact = values.astype(np.float32, copy=False)
    if not np.all(np.isfinite(compact)):
        return values
    if np.any((values != 0.0) & (compact == 0.0)):
        return values
    return compact


def _materialize_one_pass_iterables(value: object) -> object:
    """Materialize nested one-pass iterables before NumPy consumes them."""

    if isinstance(value, np.ndarray):
        if value.dtype != object:
            return value
        return _materialize_one_pass_iterables(value.tolist())
    if isinstance(value, (str, bytes)):
        return value
    if not isinstance(value, Iterable):
        return value
    return [_materialize_one_pass_iterables(item) for item in value]


def _contains_boolean_value(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        if value.dtype == np.bool_:
            return value.size > 0
        if value.dtype == object:
            return any(_contains_boolean_value(item) for item in value.ravel(order="C"))
        return False
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, np.generic):
        return isinstance(value.item(), (bool, np.bool_))
    if isinstance(value, Iterable):
        return any(_contains_boolean_value(item) for item in value)
    return False


def _matrix(values, *, name: str) -> np.ndarray:
    materialized = _materialize_one_pass_iterables(values)
    if _contains_boolean_value(materialized):
        raise ValueError(f"{name} must contain numeric feature values, not boolean flags.")
    matrix = np.asarray(materialized, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0 or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite non-empty matrix.")
    return matrix


def _finite_scalar_float(value, *, name: str) -> float:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar.") from exc
    if array.shape != ():
        raise ValueError(f"{name} must be a finite scalar.")
    scalar = array.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite scalar.")
    try:
        parsed = float(scalar)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar.") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be a finite scalar.")
    return parsed


def _nonnegative_float(value, *, name: str) -> float:
    parsed = _finite_scalar_float(value, name=name)
    if parsed < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return parsed


def _positive_int(value, *, name: str) -> int:
    parsed = _finite_scalar_float(value, name=name)
    if parsed < 1.0 or not parsed.is_integer():
        raise ValueError(f"{name} must be a positive integer.")
    return int(parsed)
