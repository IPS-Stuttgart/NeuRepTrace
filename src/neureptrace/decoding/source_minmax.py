"""Strict source-only min-max scaling helper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_MINMAX_PROTOCOL = "strict_source_only_minmax_scaling"
SOURCE_MINMAX_CATEGORY = "1_strict_source_only"
_RANGE_ERROR = "feature_range must contain exactly two finite numeric non-boolean values with low < high."


@dataclass(frozen=True, slots=True)
class SourceMinMaxReference:
    minimum: np.ndarray
    maximum: np.ndarray
    feature_range: tuple[float, float]
    n_fit_rows: int


@dataclass(frozen=True, slots=True)
class SourceMinMaxResult:
    train_features: np.ndarray
    test_features: np.ndarray
    reference: SourceMinMaxReference
    metadata: dict[str, object] = field(default_factory=dict)


def fit_source_minmax_transform(*, source_features, test_features, feature_range=(0.0, 1.0)) -> SourceMinMaxResult:
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
    return SourceMinMaxResult(train_features=train, test_features=test_out, reference=reference, metadata=metadata)


def fit_source_minmax_reference(source_features, *, feature_range=(0.0, 1.0)) -> SourceMinMaxReference:
    source = _matrix(source_features, name="source_features")
    low, high = _range(feature_range)
    return SourceMinMaxReference(minimum=np.min(source, axis=0), maximum=np.max(source, axis=0), feature_range=(low, high), n_fit_rows=int(source.shape[0]))


def apply_source_minmax_transform(features, reference: SourceMinMaxReference) -> np.ndarray:
    matrix = _matrix(features, name="features")
    if matrix.shape[1] != reference.minimum.shape[0]:
        raise ValueError("features width must match source minmax reference.")
    denom = np.maximum(reference.maximum - reference.minimum, np.finfo(float).eps)
    low, high = reference.feature_range
    scaled = (matrix - reference.minimum) / denom
    return (scaled * (high - low) + low).astype(np.float32, copy=False)


def _matrix(values, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values.")
    return matrix


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
