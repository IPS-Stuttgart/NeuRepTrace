"""Fixed absolute-value feature transform."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

ABS_FEATURE_PROTOCOL = "fixed_absolute_value_transform"
ABS_FEATURE_CATEGORY = "1_strict_source_only_compatible"


def _materialize_nested_iterables(value: object) -> object:
    """Materialize one-pass feature containers before numeric conversion."""

    if isinstance(value, np.ndarray):
        if value.dtype != object:
            return value
        return [_materialize_nested_iterables(item) for item in value.tolist()]
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        return value
    return [_materialize_nested_iterables(item) for item in value]


def _contains_boolean(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.bool_):
            return True
        if value.dtype != object:
            return False
        return any(_contains_boolean(item) for item in value.ravel())
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        return False
    return any(_contains_boolean(item) for item in value)


def absolute_value_features(features):
    """Return absolute-valued feature rows."""

    raw_features = _materialize_nested_iterables(features)
    if _contains_boolean(raw_features):
        raise ValueError("features must contain numeric values, not boolean flags")
    try:
        matrix = np.asarray(raw_features, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("features must be a two-dimensional numeric matrix") from exc
    if matrix.ndim != 2:
        raise ValueError("features must be two-dimensional")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features must be finite")
    metadata = {
        "abs_feature_protocol": ABS_FEATURE_PROTOCOL,
        "abs_feature_protocol_category": ABS_FEATURE_CATEGORY,
        "abs_feature_has_fitted_parameters": False,
        "abs_feature_uses_labels": False,
        "abs_feature_valid_for_strict_source_only": True,
        "abs_feature_n_rows": int(matrix.shape[0]),
        "abs_feature_dim": int(matrix.shape[1]),
    }
    return np.abs(matrix).astype(np.float32, copy=False), metadata
