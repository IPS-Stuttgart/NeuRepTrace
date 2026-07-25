from __future__ import annotations

import numpy as np

from neureptrace.decoding.row_normalization import normalize_source_and_test_rows


def test_row_normalization_preserves_tiny_nonzero_components_and_norms() -> None:
    source = np.asarray([[1.0e100, 1.0e50]], dtype=float)
    test = np.asarray([[1.0e80, 1.0e30]], dtype=float)

    with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
        result = normalize_source_and_test_rows(
            source_features=source,
            test_features=test,
        )

    assert result.train_features.dtype == np.float64
    assert result.test_features.dtype == np.float64
    np.testing.assert_allclose(result.train_features, [[1.0, 1.0e-50]], rtol=1.0e-15, atol=0.0)
    np.testing.assert_allclose(result.test_features, [[1.0, 1.0e-50]], rtol=1.0e-15, atol=0.0)
    np.testing.assert_allclose(result.train_norms, [1.0e100], rtol=1.0e-15)
    np.testing.assert_allclose(result.test_norms, [1.0e80], rtol=1.0e-15)
    assert np.isfinite(result.train_norms).all()
    assert np.isfinite(result.test_norms).all()


def test_row_normalization_keeps_compact_features_for_representable_values() -> None:
    result = normalize_source_and_test_rows(
        source_features=[[3.0, 4.0]],
        test_features=[[5.0, 12.0]],
    )

    assert result.train_features.dtype == np.float32
    assert result.test_features.dtype == np.float32
    assert result.train_norms.dtype == np.float64
    assert result.test_norms.dtype == np.float64
