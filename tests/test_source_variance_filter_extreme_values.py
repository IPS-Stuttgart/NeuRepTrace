from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_variance_filter import fit_source_variance_filter, source_feature_variances


def test_source_feature_variances_handles_extreme_constant_columns() -> None:
    source = np.asarray([[1e308, 1.0], [1e308, 3.0]], dtype=float)

    with np.errstate(over="raise", invalid="raise"):
        variances = source_feature_variances(source)

    np.testing.assert_allclose(variances, [0.0, 2.0])
    assert np.all(np.isfinite(variances))


def test_source_variance_filter_preserves_large_finite_outputs() -> None:
    source = np.asarray([[1e40, 0.0], [2e40, 1.0]], dtype=float)
    test = np.asarray([[1.5e40, 0.5]], dtype=float)

    with np.errstate(over="raise", under="raise", invalid="raise"):
        result = fit_source_variance_filter(source_features=source, test_features=test)

    assert result.train_features.dtype == np.float64
    assert result.test_features.dtype == np.float64
    assert result.variances.dtype == np.float64
    assert np.all(np.isfinite(result.train_features))
    assert np.all(np.isfinite(result.test_features))
    assert np.all(np.isfinite(result.variances))
    np.testing.assert_allclose(result.train_features, source)
    np.testing.assert_allclose(result.test_features, test)
    np.testing.assert_allclose(result.variances, [5e79, 0.5])


def test_source_variance_filter_preserves_tiny_nonzero_outputs() -> None:
    source = np.asarray([[1e-50, 0.0], [2e-50, 1.0]], dtype=float)
    test = np.asarray([[1.5e-50, 0.5]], dtype=float)

    result = fit_source_variance_filter(source_features=source, test_features=test)

    assert result.train_features.dtype == np.float64
    assert result.test_features.dtype == np.float64
    assert result.variances.dtype == np.float64
    assert np.all(result.train_features[:, 0] != 0.0)
    assert result.test_features[0, 0] != 0.0
    assert result.variances[0] != 0.0
