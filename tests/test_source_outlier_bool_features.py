from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_outlier import compute_source_outlier_weights


def test_source_outlier_rejects_boolean_numpy_feature_matrix() -> None:
    with pytest.raises(ValueError, match="source_features"):
        compute_source_outlier_weights(np.asarray([[True, False], [False, True]], dtype=bool), ["a", "b"])


def test_source_outlier_rejects_boolean_sequence_feature_matrix() -> None:
    with pytest.raises(ValueError, match="source_features"):
        compute_source_outlier_weights([[0.0, True], [1.0, False]], ["a", "b"])


def test_source_outlier_accepts_numeric_zero_one_feature_values() -> None:
    result = compute_source_outlier_weights([[0, 1], [1, 0]], ["a", "b"])

    assert result.sample_weights.shape == (2,)
