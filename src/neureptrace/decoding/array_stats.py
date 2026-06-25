"""Small array summary helpers for decoding diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

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

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("values must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("values must be finite.")
    if scale_floor <= 0.0 or not np.isfinite(scale_floor):
        raise ValueError("scale_floor must be positive and finite.")
    mean = np.mean(matrix, axis=0)
    scale = np.maximum(np.std(matrix, axis=0, ddof=1 if matrix.shape[0] > 1 else 0), float(scale_floor))
    minimum = np.min(matrix, axis=0)
    maximum = np.max(matrix, axis=0)
    return ArrayStatsResult(
        mean=mean.astype(np.float32, copy=False),
        scale=scale.astype(np.float32, copy=False),
        minimum=minimum.astype(np.float32, copy=False),
        maximum=maximum.astype(np.float32, copy=False),
        metadata={"array_stats_rows": int(matrix.shape[0]), "array_stats_columns": int(matrix.shape[1])},
    )
