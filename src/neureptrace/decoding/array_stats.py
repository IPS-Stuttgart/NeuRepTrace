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
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("values must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("values must be finite.")
    mean = np.mean(matrix, axis=0)
    scale = np.maximum(np.std(matrix, axis=0, ddof=1 if matrix.shape[0] > 1 else 0), floor)
    minimum = np.min(matrix, axis=0)
    maximum = np.max(matrix, axis=0)
    return ArrayStatsResult(
        mean=mean.astype(np.float32, copy=False),
        scale=scale.astype(np.float32, copy=False),
        minimum=minimum.astype(np.float32, copy=False),
        maximum=maximum.astype(np.float32, copy=False),
        metadata={"array_stats_rows": int(matrix.shape[0]), "array_stats_columns": int(matrix.shape[1])},
    )


def _positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be positive and finite.")
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"{name} must be positive and finite.")
        value = value.item()
    if isinstance(value, (list, tuple, dict, set)):
        raise ValueError(f"{name} must be positive and finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite.") from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed
