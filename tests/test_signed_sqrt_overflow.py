from __future__ import annotations

import numpy as np

from neureptrace.decoding.signed_sqrt import signed_sqrt_transform, transform_train_test_signed_sqrt


def test_signed_sqrt_avoids_intermediate_scale_overflow() -> None:
    transformed = signed_sqrt_transform([[1e308, -1e308]], scale=1e-308)

    assert np.isfinite(transformed).all()
    np.testing.assert_allclose(transformed, np.asarray([[1e308, -1e308]]), rtol=1e-15)


def test_signed_sqrt_wrapper_preserves_values_above_float32_range() -> None:
    result = transform_train_test_signed_sqrt(
        train_features=[[1e308, -1e308]],
        test_features=[[1e308, -1e308]],
        config={"scale": 1e-308},
    )

    assert result.train_features.dtype == np.float64
    assert result.test_features.dtype == np.float64
    assert np.isfinite(result.train_features).all()
    assert np.isfinite(result.test_features).all()
    np.testing.assert_allclose(result.train_features, np.asarray([[1e308, -1e308]]), rtol=1e-15)
    np.testing.assert_allclose(result.test_features, np.asarray([[1e308, -1e308]]), rtol=1e-15)


def test_signed_sqrt_wrapper_preserves_nonzero_values_below_float32_range() -> None:
    result = transform_train_test_signed_sqrt(
        train_features=[[1e-308, -1e-308]],
        test_features=[[1e-308, -1e-308]],
    )

    assert result.train_features.dtype == np.float64
    assert result.test_features.dtype == np.float64
    assert np.count_nonzero(result.train_features) == 2
    assert np.count_nonzero(result.test_features) == 2
    np.testing.assert_allclose(result.train_features, np.asarray([[1e-154, -1e-154]]), rtol=1e-15)
    np.testing.assert_allclose(result.test_features, np.asarray([[1e-154, -1e-154]]), rtol=1e-15)
