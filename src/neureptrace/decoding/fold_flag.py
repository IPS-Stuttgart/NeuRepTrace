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


def _contains_complex(value: object) -> bool:
    """Return whether a materialized feature container has complex values."""

    if isinstance(value, (complex, np.complexfloating)):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.complexfloating):
            return bool(value.size)
        if value.dtype != object:
            return False
        return any(_contains_complex(item) for item in value.ravel())
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        return False
    return any(_contains_complex(item) for item in value)


def absolute_value_features(features):
    """Return absolute-valued feature rows."""

    raw_features = _materialize_nested_iterables(features)
    if _contains_boolean(raw_features):
        raise ValueError("features must contain numeric values, not boolean flags")
    if _contains_complex(raw_features):
        raise ValueError("features must contain real-valued numeric values, not complex values")
    try:
        matrix = np.asarray(raw_features, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("features must be a two-dimensional numeric matrix") from exc
    if matrix.ndim != 2:
        raise ValueError("features must be two-dimensional")
    if matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("features must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features must be finite")

    absolute_matrix = np.abs(matrix)
    if np.any(absolute_matrix > np.finfo(np.float32).max):
        raise ValueError("features must be representable as finite float32 values")

    metadata = {
        "abs_feature_protocol": ABS_FEATURE_PROTOCOL,
        "abs_feature_protocol_category": ABS_FEATURE_CATEGORY,
        "abs_feature_has_fitted_parameters": False,
        "abs_feature_uses_labels": False,
        "abs_feature_valid_for_strict_source_only": True,
        "abs_feature_n_rows": int(matrix.shape[0]),
        "abs_feature_dim": int(matrix.shape[1]),
    }
    return absolute_matrix.astype(np.float32, copy=False), metadata
