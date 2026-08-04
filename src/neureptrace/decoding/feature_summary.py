"""Feature summary utilities for decoding diagnostics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class FeatureSummaryResult:
    """Column-wise feature summaries."""

    mean: np.ndarray
    scale: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def summarize_features(features: Iterable[Iterable[float]] | np.ndarray, *, ddof: int | str = 1) -> FeatureSummaryResult:
    """Return simple column-wise summary statistics for a feature matrix."""

    matrix = _feature_matrix(features)
    parsed_ddof = _nonnegative_int(ddof, name="ddof")
    if parsed_ddof >= matrix.shape[0]:
        parsed_ddof = 0
    mean, scale = _stable_mean_and_std(matrix, axis=0, ddof=parsed_ddof)
    minimum = np.min(matrix, axis=0)
    maximum = np.max(matrix, axis=0)
    global_mean, global_scale = _stable_mean_and_std(matrix, axis=None, ddof=parsed_ddof)
    metadata = {
        "feature_summary": True,
        "feature_summary_n_rows": int(matrix.shape[0]),
        "feature_summary_n_features": int(matrix.shape[1]),
        "feature_summary_ddof": int(parsed_ddof),
        "feature_summary_global_mean": float(global_mean),
        "feature_summary_global_scale": float(global_scale),
    }
    return FeatureSummaryResult(
        mean=mean,
        scale=scale,
        minimum=minimum,
        maximum=maximum,
        metadata=metadata,
    )


def _stable_mean_and_std(values: np.ndarray, *, axis: int | None, ddof: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute finite moments without overflowing intermediate reductions."""

    magnitude = np.max(np.abs(values), axis=axis, keepdims=True)
    normalized = np.zeros_like(values)
    np.divide(values, magnitude, out=normalized, where=magnitude > 0.0)
    normalized_mean = np.mean(normalized, axis=axis)
    normalized_std = np.std(normalized, axis=axis, ddof=ddof)
    rescale = np.squeeze(magnitude) if axis is None else np.squeeze(magnitude, axis=axis)
    return _rescale_finite(normalized_mean, rescale), _rescale_finite(normalized_std, rescale)


def _rescale_finite(values: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Rescale normalized statistics, saturating unrepresentable magnitudes."""

    value_array, scale_array = np.broadcast_arrays(
        np.asarray(values, dtype=float),
        np.asarray(scales, dtype=float),
    )
    output = np.zeros_like(value_array)
    active = (value_array != 0.0) & (scale_array != 0.0)
    maximum = np.finfo(float).max
    limits = np.full_like(scale_array, maximum)
    large_scale = scale_array > 1.0
    limits[large_scale] = maximum / scale_array[large_scale]
    safe = active & (np.abs(value_array) <= limits)
    np.multiply(value_array, scale_array, out=output, where=safe)
    overflow = active & ~safe
    output[overflow] = np.copysign(maximum, value_array[overflow])
    return output


def _materialize_nested_iterables(value: object) -> object:
    """Materialize one-pass feature containers before validation and conversion."""

    if isinstance(value, np.ndarray):
        if value.dtype != object:
            return value
        return [_materialize_nested_iterables(item) for item in value.tolist()]
    if hasattr(value, "__array__"):
        return value
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        return value
    return [_materialize_nested_iterables(item) for item in value]


def _contains_boolean(value: object) -> bool:
    """Return whether a feature container contains Boolean-valued entries."""

    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.bool_):
            return bool(value.size)
        if value.dtype == object:
            return any(_contains_boolean(item) for item in value.ravel(order="C"))
        return False
    if hasattr(value, "__array__"):
        try:
            return _contains_boolean(np.asarray(value))
        except (TypeError, ValueError):
            return False
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, Sequence):
        return any(_contains_boolean(item) for item in value)
    return False


def _contains_complex(value: object) -> bool:
    """Return whether a feature container contains complex-valued entries."""

    if isinstance(value, (complex, np.complexfloating)):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.complexfloating):
            return bool(value.size)
        if value.dtype == object:
            return any(_contains_complex(item) for item in value.ravel(order="C"))
        return False
    if hasattr(value, "__array__"):
        try:
            return _contains_complex(np.asarray(value))
        except (TypeError, ValueError):
            return False
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, Sequence):
        return any(_contains_complex(item) for item in value)
    return False


def _feature_matrix(values: Iterable[Iterable[float]] | np.ndarray) -> np.ndarray:
    materialized = _materialize_nested_iterables(values)
    if _contains_boolean(materialized):
        raise ValueError("features must contain numeric values, not boolean flags.")
    if _contains_complex(materialized):
        raise ValueError("features must contain real-valued values.")
    try:
        matrix = np.asarray(materialized, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("features must be a non-empty two-dimensional numeric matrix.") from exc
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("features must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features must contain finite values.")
    return matrix


def _nonnegative_int(value: int | str, *, name: str) -> int:
    message = f"{name} must be a non-negative integer."
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        value = value.item()
    if isinstance(value, (list, tuple, dict, set)):
        raise ValueError(message)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 0:
        raise ValueError(message)
    return int(parsed)
