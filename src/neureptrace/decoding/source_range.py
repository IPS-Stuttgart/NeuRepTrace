"""Strict source-only feature range helper."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

SOURCE_RANGE_CATEGORY = "1_strict_source_only"
SOURCE_RANGE_PROTOCOL = "strict_source_only_feature_range"


def source_feature_range(source_features):
    """Return feature-wise minimum and maximum from source rows only."""
    matrix = _feature_matrix(source_features, name="source_features")
    return np.min(matrix, axis=0), np.max(matrix, axis=0)


def apply_source_range_clip(features, lower, upper):
    """Clip rows with precomputed source-only feature ranges."""
    matrix = _feature_matrix(features, name="features")
    lo = _bound_vector(lower, name="lower")
    hi = _bound_vector(upper, name="upper")
    if matrix.shape[1] != lo.shape[0] or matrix.shape[1] != hi.shape[0]:
        raise ValueError("features width must match lower and upper bounds.")
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        raise ValueError("lower and upper bounds must contain finite values.")
    if np.any(lo > hi):
        raise ValueError("lower bounds must not exceed upper bounds.")
    clipped = np.minimum(np.maximum(matrix, lo), hi)
    return _compact_float32(clipped), clipped != matrix


def source_range_clip(*, source_features, test_features):
    """Fit source range bounds and clip source/test rows."""
    source = _feature_matrix(source_features, name="source_features")
    test = _feature_matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
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
        "source_range_valid_for_strict_source_only": True,
        "source_range_valid_for_benchmark": True,
        "source_range_n_source_rows": int(source.shape[0]),
        "source_range_n_test_rows": int(test.shape[0]),
        "source_range_feature_dim": int(source.shape[1]),
        "source_range_train_values_clipped": int(np.count_nonzero(train_mask)),
        "source_range_test_values_clipped": int(np.count_nonzero(test_mask)),
    }
    return train, test_out, lower, upper, train_mask, test_mask, metadata


def _compact_float32(values: np.ndarray) -> np.ndarray:
    """Use float32 only when conversion preserves finite, nonzero values."""

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        compact = values.astype(np.float32, copy=False)
    if not np.all(np.isfinite(compact)):
        return values
    if np.any((values != 0.0) & (compact == 0.0)):
        return values
    return compact


def _materialize_one_pass_iterables(value):
    """Materialize nested one-pass iterables before NumPy conversion."""

    if isinstance(value, np.ndarray):
        if value.dtype != object:
            return value
        return _materialize_one_pass_iterables(value.tolist())
    if isinstance(value, (str, bytes)):
        return value
    if hasattr(value, "__array__"):
        return value
    if not isinstance(value, Iterable):
        return value
    return [_materialize_one_pass_iterables(item) for item in value]


def _contains_boolean_values(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        if value.dtype == np.bool_:
            return value.size > 0
        if value.dtype == object:
            return any(_contains_boolean_values(item) for item in value.flat)
        return False
    if isinstance(value, (str, bytes)):
        return False
    if hasattr(value, "__array__"):
        try:
            return _contains_boolean_values(np.asarray(value, dtype=object))
        except (TypeError, ValueError):
            return False
    if isinstance(value, Iterable):
        return any(_contains_boolean_values(item) for item in value)
    return False


def _contains_complex_values(value) -> bool:
    if isinstance(value, (complex, np.complexfloating)):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.complexfloating):
            return value.size > 0
        if value.dtype == object:
            return any(_contains_complex_values(item) for item in value.flat)
        return False
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, np.generic):
        return isinstance(value.item(), complex)
    if hasattr(value, "__array__"):
        try:
            return _contains_complex_values(np.asarray(value, dtype=object))
        except (TypeError, ValueError):
            return False
    if isinstance(value, Iterable):
        return any(_contains_complex_values(item) for item in value)
    return False


def _feature_matrix(values, *, name: str):
    materialized = _materialize_one_pass_iterables(values)
    if _contains_boolean_values(materialized):
        raise ValueError(f"{name} must contain numeric, non-boolean values.")
    if _contains_complex_values(materialized):
        raise ValueError(f"{name} must contain real-valued values, not complex values.")
    matrix = np.asarray(materialized, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values.")
    return matrix


def _bound_vector(values, *, name: str):
    materialized = _materialize_one_pass_iterables(values)
    if _contains_boolean_values(materialized):
        raise ValueError(f"{name} must contain numeric, non-boolean values.")
    if _contains_complex_values(materialized):
        raise ValueError(f"{name} must contain real-valued values, not complex values.")
    return np.asarray(materialized, dtype=float).reshape(-1)
