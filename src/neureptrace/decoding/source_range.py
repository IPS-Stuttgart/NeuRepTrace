"""Strict source-only feature range helper."""

from __future__ import annotations

import numpy as np

SOURCE_RANGE_CATEGORY = "1_strict_source_only"
SOURCE_RANGE_PROTOCOL = "strict_source_only_feature_range"


def source_feature_range(source_features):
    """Return feature-wise minimum and maximum from source rows only."""
    matrix = np.asarray(source_features, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("source_features must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("source_features must contain finite values.")
    return np.min(matrix, axis=0), np.max(matrix, axis=0)


def apply_source_range_clip(features, lower, upper):
    """Clip rows with precomputed source-only feature ranges."""
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("features must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features must contain finite values.")
    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)
    if matrix.shape[1] != lo.shape[0] or matrix.shape[1] != hi.shape[0]:
        raise ValueError("features width must match lower and upper bounds.")
    clipped = np.minimum(np.maximum(matrix, lo), hi)
    return clipped.astype(np.float32, copy=False), clipped != matrix


def source_range_clip(*, source_features, test_features):
    """Fit source range bounds and clip source/test rows."""
    source = np.asarray(source_features, dtype=float)
    test = np.asarray(test_features, dtype=float)
    if source.ndim != 2 or test.ndim != 2 or source.shape[1] != test.shape[1]:
        raise ValueError("source_features and test_features must be two-dimensional matrices with the same feature width.")
    lower, upper = source_feature_range(source)
    train, train_mask = apply_source_range_clip(source, lower, upper)
    test_out, test_mask = apply_source_range_clip(test, lower, upper)
    metadata = {
        "source_range_protocol": SOURCE_RANGE_PROTOCOL,
        "source_range_protocol_category": SOURCE_RANGE_CATEGORY,
        "source_range_uses_source_features": True,
        "source_range_uses_test_features_for_fitting": False,
        "source_range_uses_test_labels": False,
        "source_range_n_source_rows": int(source.shape[0]),
        "source_range_n_test_rows": int(test.shape[0]),
        "source_range_feature_dim": int(source.shape[1]),
        "source_range_train_values_clipped": int(np.count_nonzero(train_mask)),
        "source_range_test_values_clipped": int(np.count_nonzero(test_mask)),
    }
    return train, test_out, lower, upper, train_mask, test_mask, metadata
