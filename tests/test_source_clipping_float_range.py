from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_clipping import fit_source_feature_clipping


@pytest.mark.parametrize("scale", [1.0e100, 1.0e-50])
def test_source_clipping_preserves_values_outside_float32_range(scale: float) -> None:
    source = np.asarray([[scale], [2.0 * scale]], dtype=float)
    test = np.asarray([[1.5 * scale]], dtype=float)

    result = fit_source_feature_clipping(
        source_features=source,
        test_features=test,
        config={"lower_quantile": 0.0, "upper_quantile": 1.0},
    )

    assert result.train_features.dtype == np.float64
    assert result.test_features.dtype == np.float64
    assert result.lower_bounds.dtype == np.float64
    assert result.upper_bounds.dtype == np.float64
    np.testing.assert_array_equal(result.train_features, source)
    np.testing.assert_array_equal(result.test_features, test)
    np.testing.assert_array_equal(result.lower_bounds, source[0])
    np.testing.assert_array_equal(result.upper_bounds, source[1])
    assert np.isfinite(result.train_features).all()
    assert np.count_nonzero(result.train_features) == result.train_features.size


def test_source_clipping_keeps_float32_for_representable_values() -> None:
    result = fit_source_feature_clipping(
        source_features=[[0.0], [1.0], [2.0]],
        test_features=[[1.5]],
        config={"lower_quantile": 0.0, "upper_quantile": 1.0},
    )

    assert result.train_features.dtype == np.float32
    assert result.test_features.dtype == np.float32
    assert result.lower_bounds.dtype == np.float32
    assert result.upper_bounds.dtype == np.float32
