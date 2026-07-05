"""Fixed absolute-value feature transform."""

from __future__ import annotations

import numpy as np

ABS_FEATURE_PROTOCOL = "fixed_absolute_value_transform"
ABS_FEATURE_CATEGORY = "1_strict_source_only_compatible"


def absolute_value_features(features):
    """Return absolute-valued feature rows."""

    matrix = np.asarray(features, dtype=float)
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
