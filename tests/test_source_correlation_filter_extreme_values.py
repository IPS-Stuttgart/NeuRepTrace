from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_correlation_filter import (
    DEFAULT_EPSILON,
    fit_source_correlation_filter,
    source_feature_correlation,
    source_feature_importance,
)


def test_source_correlation_filter_handles_extreme_finite_features() -> None:
    maximum = np.finfo(np.float64).max
    source = np.asarray(
        [
            [maximum, maximum, 0.0],
            [maximum, -maximum, 0.0],
        ]
    )
    test = np.asarray([[maximum, 0.0, 0.0]])

    with np.errstate(over="raise", divide="raise", invalid="raise", under="raise"):
        correlation = source_feature_correlation(source)
        importance = source_feature_importance(source)
        result = fit_source_correlation_filter(source_features=source, test_features=test)

    assert np.all(np.isfinite(correlation))
    np.testing.assert_allclose(correlation, np.eye(3))
    assert importance[0] == pytest.approx(DEFAULT_EPSILON)
    assert importance[1] == np.finfo(np.float64).max
    assert importance[2] == pytest.approx(DEFAULT_EPSILON)

    assert np.all(np.isfinite(result.train_features))
    assert np.all(np.isfinite(result.test_features))
    assert np.all(np.isfinite(result.correlation))
    assert np.all(np.isfinite(result.importance))
    assert result.train_features.dtype == np.float64
    assert result.test_features.dtype == np.float64
