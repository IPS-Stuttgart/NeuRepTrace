from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_threshold import fit_source_threshold_transform


def test_source_threshold_preserves_values_outside_float32_range() -> None:
    with np.errstate(over="raise", under="raise", invalid="raise"):
        result = fit_source_threshold_transform(
            source_features=[[-1.0], [1.0]],
            test_features=[[-2.0], [2.0]],
            config={
                "threshold_mode": "zero",
                "positive_value": 1.0e40,
                "negative_value": 1.0e-50,
            },
        )

    assert result.train_features.dtype == np.float64
    assert result.test_features.dtype == np.float64
    np.testing.assert_array_equal(result.train_features.ravel(), [1.0e-50, 1.0e40])
    np.testing.assert_array_equal(result.test_features.ravel(), [1.0e-50, 1.0e40])
    assert np.all(np.isfinite(result.train_features))
    assert np.all(np.isfinite(result.test_features))
    assert np.count_nonzero(result.train_features) == result.train_features.size
    assert np.count_nonzero(result.test_features) == result.test_features.size


def test_source_threshold_keeps_float32_for_representable_values() -> None:
    result = fit_source_threshold_transform(
        source_features=[[-1.0], [1.0]],
        test_features=[[-2.0], [2.0]],
        config={"threshold_mode": "zero", "positive_value": 2.0, "negative_value": -3.0},
    )

    assert result.train_features.dtype == np.float32
    assert result.test_features.dtype == np.float32
