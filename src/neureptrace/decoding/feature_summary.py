"""Feature summary utilities for decoding diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
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


def summarize_features(features: Sequence[Sequence[float]] | np.ndarray, *, ddof: int | str = 1) -> FeatureSummaryResult:
    """Return simple column-wise summary statistics for a feature matrix."""

    matrix = _feature_matrix(features)
    parsed_ddof = _nonnegative_int(ddof, name="ddof")
    if parsed_ddof >= matrix.shape[0]:
        parsed_ddof = 0
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0, ddof=parsed_ddof)
    minimum = np.min(matrix, axis=0)
    maximum = np.max(matrix, axis=0)
    metadata = {
        "feature_summary": True,
        "feature_summary_n_rows": int(matrix.shape[0]),
        "feature_summary_n_features": int(matrix.shape[1]),
        "feature_summary_ddof": int(parsed_ddof),
        "feature_summary_global_mean": float(np.mean(matrix)),
        "feature_summary_global_scale": float(np.std(matrix, ddof=parsed_ddof)) if matrix.size > parsed_ddof else 0.0,
    }
    return FeatureSummaryResult(
        mean=mean.astype(np.float32, copy=False),
        scale=scale.astype(np.float32, copy=False),
        minimum=minimum.astype(np.float32, copy=False),
        maximum=maximum.astype(np.float32, copy=False),
        metadata=metadata,
    )


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("features must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features must contain finite values.")
    return matrix


def _nonnegative_int(value: int | str, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer.")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(parsed)
