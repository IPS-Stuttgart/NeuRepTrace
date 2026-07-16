from __future__ import annotations

import numpy as np

from neureptrace.decoding.row_l1 import normalize_rows_l1, normalize_train_test_rows_l1


def test_train_test_row_l1_preserves_components_that_float32_would_underflow() -> None:
    train = np.asarray([[1.0, 1e-50]], dtype=float)
    test = np.asarray([[-1e-50, 1.0]], dtype=float)
    expected_train, _ = normalize_rows_l1(train)
    expected_test, _ = normalize_rows_l1(test)

    with np.errstate(over="raise", under="raise", invalid="raise"):
        result = normalize_train_test_rows_l1(train_features=train, test_features=test)

    assert result.train_features.dtype == np.float64
    assert result.test_features.dtype == np.float64
    np.testing.assert_array_equal(result.train_features, expected_train)
    np.testing.assert_array_equal(result.test_features, expected_test)
    assert result.train_features[0, 1] != 0.0
    assert result.test_features[0, 0] != 0.0


def test_train_test_row_l1_keeps_float32_for_representable_outputs() -> None:
    result = normalize_train_test_rows_l1(
        train_features=[[1.0, -2.0]],
        test_features=[[3.0, 4.0]],
    )

    assert result.train_features.dtype == np.float32
    assert result.test_features.dtype == np.float32
