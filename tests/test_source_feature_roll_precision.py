from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_roll import augment_source_with_feature_roll, roll_feature_row


def test_roll_feature_row_preserves_extreme_finite_values() -> None:
    with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
        large = roll_feature_row([1.0e40, -2.0e40], shift=1)
        tiny = roll_feature_row([1.0e-100, 2.0e-100], shift=1)

    assert large.dtype == np.float64
    assert tiny.dtype == np.float64
    np.testing.assert_allclose(large, [-2.0e40, 1.0e40])
    np.testing.assert_allclose(tiny, [2.0e-100, 1.0e-100])
    assert np.all(np.isfinite(large))
    assert np.all(tiny != 0.0)


def test_source_roll_disabled_preserves_extreme_finite_features() -> None:
    features = np.asarray(
        [
            [1.0e40, 2.0e40],
            [1.0e-100, 2.0e-100],
        ],
        dtype=float,
    )

    with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
        result = augment_source_with_feature_roll(features, [0, 1])

    assert result.features.dtype == np.float64
    np.testing.assert_allclose(result.features, features)
    assert np.all(np.isfinite(result.features))
    assert np.all(result.features[1] != 0.0)


def test_source_roll_synthetic_rows_preserve_large_finite_features() -> None:
    features = np.asarray(
        [
            [1.0e40, 2.0e40],
            [3.0e40, 4.0e40],
        ],
        dtype=float,
    )

    with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
        result = augment_source_with_feature_roll(
            features,
            [0, 1],
            config={
                "synthetic_per_class": 1,
                "max_shift": 1,
                "preserve_original": False,
                "random_state": 7,
            },
        )

    assert result.features.dtype == np.float64
    np.testing.assert_allclose(result.features, [[2.0e40, 1.0e40], [4.0e40, 3.0e40]])
    assert np.all(np.isfinite(result.features))


def test_source_roll_keeps_float32_for_representable_features() -> None:
    assert roll_feature_row([1.0, 2.0], shift=1).dtype == np.float32

    result = augment_source_with_feature_roll([[1.0, 2.0]], [0])

    assert result.features.dtype == np.float32
