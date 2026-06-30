"""Source-only range selector."""

from __future__ import annotations

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
    source = _matrix(source_features)
    test = _matrix(test_features)
    if source.shape[1] != test.shape[1]:
        raise ValueError("feature widths differ")
    min_range_value = _nonnegative_float(min_range, name="min_range")
    top_k_value = None if top_k is None else _positive_int(top_k, name="top_k")
    ranges = np.ptp(source, axis=0).astype(float, copy=False)
    selected = select_source_range_features(ranges, min_range=min_range_value, top_k=top_k_value)
    return SourceRangeSelectorResult(
        train_features=source[:, selected].astype(np.float32, copy=False),
        test_features=test[:, selected].astype(np.float32, copy=False),
        selected_indices=selected.astype(int, copy=False),
        ranges=ranges.astype(np.float32, copy=False),
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
    values = np.asarray(ranges, dtype=float).reshape(-1)
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


def _matrix(values) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0 or not np.all(np.isfinite(matrix)):
        raise ValueError("expected finite non-empty matrix")
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
