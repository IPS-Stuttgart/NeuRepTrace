"""Small array summary helpers for decoding diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class ArrayStatsResult:
    """Column-wise summary statistics."""

    mean: np.ndarray
    scale: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)


def column_stats(values: Sequence[Sequence[float]] | np.ndarray, *, scale_floor: float = 1e-12) -> ArrayStatsResult:
    """Return column-wise mean, standard deviation, minimum, and maximum."""

    floor = _positive_float(scale_floor, name="scale_floor")
    if _contains_complex(values):
        raise ValueError("values must contain real-valued entries.")
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("values must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("values must be finite.")
    mean, scale = _stable_column_mean_and_std(matrix)
    scale = np.maximum(scale, floor)
    minimum = np.min(matrix, axis=0)
    maximum = np.max(matrix, axis=0)
    return ArrayStatsResult(
        mean=mean,
        scale=scale,
        minimum=minimum,
        maximum=maximum,
        metadata={"array_stats_rows": int(matrix.shape[0]), "array_stats_columns": int(matrix.shape[1])},
    )


def _contains_complex(value: object) -> bool:
    """Return whether an array-like input contains complex-valued entries."""

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


def _stable_column_mean_and_std(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute column moments without overflowing intermediate sums or squares.

    A finite sample standard deviation can exceed the largest representable
    ``float64`` value even when every input is finite. Saturate that
    unrepresentable result at the largest finite value instead of returning
    ``inf`` or raising under strict NumPy error handling.
    """

    magnitude = np.max(np.abs(matrix), axis=0)
    normalized = np.zeros_like(matrix)
    nonzero = magnitude > 0.0
    normalized[:, nonzero] = matrix[:, nonzero] / magnitude[nonzero]
    ddof = 1 if matrix.shape[0] > 1 else 0
    mean = np.mean(normalized, axis=0) * magnitude
    normalized_scale = np.std(normalized, axis=0, ddof=ddof)

    max_float = np.finfo(matrix.dtype).max
    scale_limit = np.full_like(magnitude, max_float)
    large_magnitude = magnitude > 1.0
    scale_limit[large_magnitude] = max_float / magnitude[large_magnitude]
    scale = np.minimum(normalized_scale, scale_limit) * magnitude
    return mean, scale


def _positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be positive and finite.")
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"{name} must be positive and finite.")
        value = value.item()
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be positive and finite.")
    if isinstance(value, (list, tuple, dict, set)):
        raise ValueError(f"{name} must be positive and finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite.") from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed
