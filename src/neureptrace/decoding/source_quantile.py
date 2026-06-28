"""Source-only quantile helpers."""

from __future__ import annotations

import numpy as np

SOURCE_QUANTILE_CATEGORY = "1_strict_source_only"


def source_feature_quantiles(source_features, *, lower=0.01, upper=0.99):
    matrix = np.asarray(source_features, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("source_features must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("source_features must contain finite values.")
    lo = float(lower)
    hi = float(upper)
    if not 0.0 <= lo <= hi <= 1.0:
        raise ValueError("lower and upper must satisfy 0 <= lower <= upper <= 1.")
    return np.quantile(matrix, lo, axis=0), np.quantile(matrix, hi, axis=0)
