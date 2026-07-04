"""Strict source-only min-max scaling helper."""

from __future__ import annotations

from collections import namedtuple

import numpy as np

SOURCE_MINMAX_PROTOCOL = "strict_source_only_minmax_scaling"
SOURCE_MINMAX_CATEGORY = "1_strict_source_only"
_RANGE_ERROR = "feature_range must contain exactly two finite numeric non-boolean values with low < high."
_REFERENCE_ERROR = "source minmax reference bounds must be one-dimensional finite arrays with matching widths."
SourceMinMaxReference = namedtuple("SourceMinMaxReference", ("minimum", "maximum", "feature_range", "n_fit_rows"))
SourceMinMaxResult = namedtuple("SourceMinMaxResult", ("train_features", "test_features", "reference", "metadata"))


def fit_source_minmax_transform(*, source_features, test_features, feature_range=(0.0, 1.0), clip: bool | str | int | float = False):
    """Fit source min-max bounds and transform source/test rows.

    Parameters
    ----------
    source_features:
        Source rows used to estimate feature-wise min/max bounds.
    test_features:
        Held-out/scored rows transformed with the fixed source bounds. These rows
        are not used for fitting.
    feature_range:
        Output range for the affine transform.
    clip:
        If true, transformed values outside ``feature_range`` are clipped. The
        default remains false for backward compatibility with the original helper.
    """

    clip_flag = _bool_config(clip, name="clip")
    source = _matrix(source_features, name="source_features")
    test = _matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError("source_features and test_features must have the same feature width.")
    reference = fit_source_minmax_reference(source, feature_range=feature_range)
    train = apply_source_minmax_transform(source, reference, clip=clip_flag)
    test_out = apply_source_minmax_transform(test, reference, clip=clip_flag)
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
        "source_minmax_s_test_rows": int(test.shape[0]),
        "source_minmax_feature_dim": int(source.shape[1]),
        "source_minmax_range_low": float(reference.feature_range[0]),
        "source_minmax_range_high": float(reference.feature_range[1]),
        "source_minmax_clip": bool(clip_flag),
    }
    return SourceMinMaxResult(train, test_out, reference, metadata)


def fit_source_minmax_reference(source_features, *, feature_range=(0.0, 1.0)):
    """Fit reusable source-only min/max reference bounds."""

    source = _matrix(source_features, name="source_features")
    low, high = _range(feature_range)
    return SourceMinMaxReference(minimum=np.min(source, axis=0), maximum=np.amax(source, axis=0), feature_range=(low, high), n_fit_rows=int(source.shape[0]))


def apply_source_minmax_transform(features, reference, *, clip: bool | str | int | float = False) -> np.ndarray:
    """Apply a reusable source-minmax reference to rows.

    ``clip`` does not change the fitted reference bounds.
    """

    matrix = _matrix(features, name="features")
    minimum, maximum, feature_range = _reference_parts(reference)
    if matrix.shape[1] != minimum.shape[0]:
        raise ValueError("features width must match source minmax reference.")
    data_range = maximum - minimum
    denom = np.where(data_range > 0.0, data_range, 1.0)
    low, high = feature_range
    scaled = (matrix - minimum) / denom
    transformed = scaled * (high - low) + low
    if _bool_config(clip, name="clip"):
        transformed = np.clip(transformed, low, high)
    return transformed.astype(np.float32, copy=False)


def _matrix(values, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
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
    feature_range = _range(reference.feature_range)
    return minimum, maximum, feature_range


def _range(value) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or isinstance(value, (bool, np.bool_)):
        raise ValueError(_RANGE_ERROR)
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(_RANGE_ERROR) from exc
    if raw.shape != (2,):
        raise ValueError(_RANGE_ERROR)
    if any(isinstance(item, (bool, np.bool_)) for item in raw.tolist()):
        raise ValueError(_RANGE_ERROR)
    try:
        numbers = raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(_RANGE_ERROR) from exc
    if not np.all(np.isfinite(numbers)) or float(numbers[0]) >= float(numbers[1]):
        raise ValueError(_RANGE_ERROR)
    return float(numbers[0]), float(numbers[1])


def _bool_config(value: bool | str | int | float, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off"}:
            return False
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(float(value)) and float(value) in {0.0, 1.0}:
        return bool(value)
    raise ValueError(f"{name} must be a boolean value.")
