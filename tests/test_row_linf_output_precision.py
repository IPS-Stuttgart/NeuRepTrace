from __future__ import annotations

import numpy as np

from neureptrace.decoding.row_linf import normalize_train_test_rows_linf


def test_row_linf_preserves_tiny_nonzero_normalized_components() -> None:
    with np.errstate(over="raise", under="raise", invalid="raise"):
        result = normalize_train_test_rows_linf(
            train_features=[[1.0e100, 1.0e50]],
            test_features=[[1.0e80, 1.0e30]],
        )

    assert result.train_features.dtype == np.float64
    assert result.test_features.dtype == np.float64
    assert result.train_features[0, 1] > 0.0
    assert result.test_features[0, 1] > 0.0
    np.testing.assert_allclose(
        result.train_features,
        [[1.0, 1.0e-50]],
        rtol=1.0e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.test_features,
        [[1.0, 1.0e-50]],
        rtol=1.0e-15,
        atol=0.0,
    )


def test_row_linf_keeps_float32_for_representable_values() -> None:
    result = normalize_train_test_rows_linf(
        train_features=[[2.0, 1.0]],
        test_features=[[4.0, 1.0]],
    )

    assert result.train_features.dtype == np.float32
    assert result.test_features.dtype == np.float32
