from __future__ import annotations

import numpy as np

from neureptrace.decoding.row_l2 import normalize_rows_l2, normalize_train_test_rows_l2


def test_normalize_rows_l2_avoids_overflow_for_large_finite_rows() -> None:
    features = np.asarray([[1e308, 1e308], [-1e308, 1e308]], dtype=float)

    normalized, norms = normalize_rows_l2(features)

    expected_norm = np.sqrt(2.0) * 1e308
    assert np.all(np.isfinite(norms))
    assert np.all(np.isfinite(normalized))
    np.testing.assert_allclose(norms, np.full(2, expected_norm), rtol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), np.ones(2), rtol=1e-15)


def test_train_test_row_l2_preserves_large_norms_without_float32_overflow() -> None:
    result = normalize_train_test_rows_l2(
        train_features=[[1e308, 1e308]],
        test_features=[[1e308, 0.0]],
    )

    assert result.train_norms.dtype == np.float64
    assert result.test_norms.dtype == np.float64
    assert np.all(np.isfinite(result.train_norms))
    assert np.all(np.isfinite(result.test_norms))
    np.testing.assert_allclose(result.train_norms, [np.sqrt(2.0) * 1e308], rtol=1e-15)
    np.testing.assert_allclose(result.test_norms, [1e308], rtol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(result.train_features, axis=1), [1.0], rtol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(result.test_features, axis=1), [1.0], rtol=1e-6)
