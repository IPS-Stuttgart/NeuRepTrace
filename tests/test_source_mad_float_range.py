from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_mad import fit_source_mad_transform


@pytest.mark.parametrize("magnitude", [1.0e100, 1.0e-50])
def test_source_mad_preserves_values_outside_float32_range(magnitude: float) -> None:
    source = np.asarray([[magnitude], [2.0 * magnitude]], dtype=float)
    test = np.asarray([[1.5 * magnitude]], dtype=float)

    result = fit_source_mad_transform(
        source_features=source,
        test_features=test,
        config={"center": False, "scale": False},
    )

    assert result.train_features.dtype == np.float64
    assert result.test_features.dtype == np.float64
    np.testing.assert_array_equal(result.train_features, source)
    np.testing.assert_array_equal(result.test_features, test)
    assert np.isfinite(result.train_features).all()
    assert np.count_nonzero(result.train_features) == result.train_features.size


def test_source_mad_keeps_float32_for_representable_values() -> None:
    result = fit_source_mad_transform(
        source_features=[[0.0], [1.0], [2.0]],
        test_features=[[1.5]],
        config={"center": False, "scale": False},
    )

    assert result.train_features.dtype == np.float32
    assert result.test_features.dtype == np.float32
