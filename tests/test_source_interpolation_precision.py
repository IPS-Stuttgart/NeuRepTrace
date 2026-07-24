from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_interpolation import (
    augment_source_with_interpolation,
    interpolate_rows,
)


@pytest.mark.parametrize("scale", [1.0e40, 1.0e-100])
def test_interpolate_rows_preserves_values_outside_float32_range(scale: float) -> None:
    with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
        row = interpolate_rows(
            [scale, 2.0 * scale],
            [3.0 * scale, 4.0 * scale],
            0.5,
        )

    assert row.dtype == np.float64
    np.testing.assert_allclose(row, [2.0 * scale, 3.0 * scale], rtol=1.0e-15, atol=0.0)
    assert np.isfinite(row).all()
    assert np.count_nonzero(row) == row.size


@pytest.mark.parametrize("scale", [1.0e40, 1.0e-100])
def test_disabled_source_interpolation_preserves_extreme_finite_features(scale: float) -> None:
    features = np.asarray([[scale, 2.0 * scale], [3.0 * scale, 4.0 * scale]])

    with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
        result = augment_source_with_interpolation(features, [0, 1])

    assert result.features.dtype == np.float64
    np.testing.assert_array_equal(result.features, features)
    assert np.isfinite(result.features).all()
    assert np.count_nonzero(result.features) == result.features.size


@pytest.mark.parametrize("scale", [1.0e40, 1.0e-100])
def test_synthetic_source_interpolation_preserves_extreme_finite_features(scale: float) -> None:
    source_row = np.asarray([scale, 2.0 * scale])
    features = np.vstack([source_row, source_row])

    with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
        result = augment_source_with_interpolation(
            features,
            [0, 0],
            config={
                "synthetic_per_class": 1,
                "preserve_original": False,
                "random_state": 7,
            },
        )

    assert result.features.dtype == np.float64
    np.testing.assert_allclose(result.features, source_row[None, :], rtol=1.0e-15, atol=0.0)
    assert np.isfinite(result.features).all()
    assert np.count_nonzero(result.features) == result.features.size


def test_source_interpolation_keeps_float32_for_representable_values() -> None:
    assert interpolate_rows([0.0, 1.0], [2.0, 3.0], 0.5).dtype == np.float32

    result = augment_source_with_interpolation([[0.0, 1.0], [2.0, 3.0]], [0, 1])

    assert result.features.dtype == np.float32
