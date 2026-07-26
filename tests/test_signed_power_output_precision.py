from __future__ import annotations

import numpy as np

from neureptrace.decoding.signed_power import (
    SignedPowerConfig,
    signed_power_transform,
    transform_train_test_signed_power,
)


def test_train_test_signed_power_preserves_values_outside_float32_range() -> None:
    train = np.asarray([[1e-30, -1e-30]], dtype=float)
    test = np.asarray([[1e80, -1e80]], dtype=float)
    expected_train = signed_power_transform(train, power=2.0)
    expected_test = signed_power_transform(test, power=2.0)

    with np.errstate(over="raise", under="raise", invalid="raise"):
        result = transform_train_test_signed_power(
            train_features=train,
            test_features=test,
            config=SignedPowerConfig(power=2.0),
        )

    assert result.train_features.dtype == np.float64
    assert result.test_features.dtype == np.float64
    np.testing.assert_array_equal(result.train_features, expected_train)
    np.testing.assert_array_equal(result.test_features, expected_test)
    assert np.all(result.train_features != 0.0)
    assert np.all(np.isfinite(result.test_features))


def test_train_test_signed_power_keeps_float32_for_representable_outputs() -> None:
    result = transform_train_test_signed_power(
        train_features=[[-4.0, 9.0]],
        test_features=[[16.0, -25.0]],
    )

    assert result.train_features.dtype == np.float32
    assert result.test_features.dtype == np.float32
