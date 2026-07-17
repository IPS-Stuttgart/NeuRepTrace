from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_range_selector import fit_source_range_selector


def test_source_range_selector_handles_extreme_finite_ranges_without_overflow() -> None:
    source = np.asarray(
        [
            [-1e308, 1.0],
            [1e308, 3.0],
        ],
        dtype=float,
    )
    test = np.asarray([[1e308, 2.0]], dtype=float)

    with np.errstate(over="raise", invalid="raise"):
        result = fit_source_range_selector(
            source_features=source,
            test_features=test,
        )

    assert result.selected_indices.tolist() == [0, 1]
    assert result.ranges.dtype == np.float64
    assert result.ranges[0] == np.finfo(float).max
    assert result.ranges[1] == 2.0
    assert np.all(np.isfinite(result.ranges))
    assert result.train_features.dtype == np.float64
    assert result.test_features.dtype == np.float64
    assert np.all(np.isfinite(result.train_features))
    assert np.all(np.isfinite(result.test_features))
    np.testing.assert_array_equal(result.train_features, source)
    np.testing.assert_array_equal(result.test_features, test)


def test_source_range_selector_keeps_float32_for_representable_outputs() -> None:
    result = fit_source_range_selector(
        source_features=[[0.0, 1.0], [2.0, 3.0]],
        test_features=[[1.0, 2.0]],
    )

    assert result.train_features.dtype == np.float32
    assert result.test_features.dtype == np.float32
    assert result.ranges.dtype == np.float32
