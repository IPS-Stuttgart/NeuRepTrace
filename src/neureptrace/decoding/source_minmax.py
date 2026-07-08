"""Strict source-only min-max scaling helper."""

from __future__ import annotations

from collections import namedtuple
from collections.abc import Iterable
from typing import Any

import numpy as np

SOURCE_MINMAX_PROTOCOL = "strict_source_only_minmax_scaling"
SOURCE_MINMAX_CATEGORY = "1_strict_source_only"
_RANGE_ERROR = "feature_range must contain exactly two finite numeric non-boolean values with low < high."
_REFERENCE_ERROR = "source minmax reference bounds must be one-dimensional finite arrays with matching widths."
SourceMinMaxReference = namedtuple("SourceMinMaxReference", ("minimum", "maximum", "feature_range", "n_fit_rows"))
SourceMinMaxResult = namedtuple("SourceMinMaxResult", ("train_features", "test_features", "reference", "metadata"))


def fit_source_minmax_transform(
    *,
    source_features: Iterable[Iterable[float]] | np.ndarray,
    test_features: Iterable[Iterable[float]] | np.ndarray,
    feature_range=(0.0, 1.0),
):
    source = _matrix(source_features, name="source_features")
    test = _matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError("source_features and test_features must have the same feature width.")
    reference = fit_source_minmax_reference(source, feature_range=feature_range)
    train = apply_source_minmax_transform(source, reference)
    test_out = apply_source_minmax_transform(test, reference)
    metadata = {
        "source_minmax": True,
        "source_minmax_protocol": SOURCE_MINMAX_PROTOCOL,
        "source_minmax_protocol_category": SOURCE_MINMAX_CATEGORY,
        "source_minmax_uses_source_features": True,
        "source_minmax_uses_test_features_for_fitting": False,
        "source_minmax_uses_test_labels": False,
        "source_minmax_valid_for_strict_source_only": True,
        "source_minmax_valid_for_benchmark": True,
        "source_minmax_n_source_rows": int(source.shape[0]),
        "source_minmax_n_test_rows": int(test.shape[0]),
        "source_minmax_feature_dim": int(source.shape[1]),
        "source_minmax_range_low": float(reference.feature_range[0]),
        "source_minmax_range_high": float(reference.feature_range[1]),
    }
    return SourceMinMaxResult(train, test_out, reference, metadata)


def fit_source_minmax_reference(source_features: Iterable[Iterable[float]] | np.ndarray, *, feature_range=(0.0, 1.0)):
    source = _matrix(source_features, name="source_features")
    low, high = _range(feature_range)
    return SourceMinMaxReference(minimum=np.min(source, axis=0), maximum=np.amax(source, axis=0), feature_range=(low, high), n_fit_rows=int(source.shape[0]))


def apply_source_minmax_transform(features: Iterable[Iterable[float]] | np.ndarray, reference) -> np.ndarray:
    matrix = _matrix(features, name="features")
    minimum, maximum, feature_range = _reference_parts(reference)
    if matrix.shape[1] != minimum.shape[0]:
        raise ValueError("features width must match source minmax reference.")
    data_range = maximum - minimum
    denom = np.where(data_range > 0.0, data_range, 1.0)
    low, high = feature_range
    scaled = (matrix - minimum) / denom
    return (scaled * (high - low) + low).astype(np.float32, copy=False)


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
            return True
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


def _matrix(values: Iterable[Iterable[float]] | np.ndarray, *, name: str) -> np.ndarray:
    materialized = _materialize_one_pass_iterables(values)
    if _contains_boolean_value(materialized):
        raise ValueError(f"{name} must contain numeric feature values, not boolean flags.")
    matrix = np.asarray(materialized, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values.")
    return matrix


def _reference_parts(reference) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    minimum = np.asarray(reference.minimum, dtype=float).reshape(-1)
    maximum = np.asarray(reference.maximum, dtype=float).reshape(-1)
    if minimum.shape[0] < 1 or maximum.shape[0] != minimum.shape[0]:
        raise ValueError(_REFERENCE_ERROR)
    if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
        raise ValueError(_REFERENCE_ERROR)
    if np.any(maximum < minimum):
        raise ValueError(_REFERENCE_ERROR)
    feature_range = _range(reference.feature_range)
    return minimum, maximum, feature_range


def _range(values) -> tuple[float, float]:
    if isinstance(values, (str, bytes)):
        raise ValueError(_RANGE_ERROR)
    try:
        items = list(values)
    except TypeError as exc:
        raise ValueError(_RANGE_ERROR) from exc
    if len(items) != 2:
        raise ValueError(_RANGE_ERROR)
    low = _range_endpoint(items[0])
    high = _range_endpoint(items[1])
    if low >= high:
        raise ValueError(_RANGE_ERROR)
    return low, high


def _range_endpoint(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(_RANGE_ERROR)
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(_RANGE_ERROR)
        value = value.item()
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(_RANGE_ERROR)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(_RANGE_ERROR) from exc
    if not np.isfinite(parsed):
        raise ValueError(_RANGE_ERROR)
    return parsed
