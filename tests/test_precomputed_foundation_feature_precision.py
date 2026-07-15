from __future__ import annotations

import numpy as np

from neureptrace.decoding.precomputed_foundation import (
    align_precomputed_foundation_features,
    make_precomputed_foundation_feature_table,
)


def test_precomputed_foundation_preserves_overflowing_finite_values() -> None:
    extreme = float(np.finfo(np.float32).max) * 2.0

    with np.errstate(over="raise", under="raise", invalid="raise"):
        table = make_precomputed_foundation_feature_table([[extreme, 1.0]])
        aligned = align_precomputed_foundation_features(table, [0])

    assert table.features.dtype == np.float64
    assert aligned.dtype == np.float64
    assert np.all(np.isfinite(table.features))
    assert np.all(np.isfinite(aligned))
    np.testing.assert_array_equal(table.features, [[extreme, 1.0]])
    np.testing.assert_array_equal(aligned, table.features)


def test_precomputed_foundation_preserves_underflowing_nonzero_values() -> None:
    tiny = np.nextafter(0.0, 1.0)

    with np.errstate(over="raise", under="raise", invalid="raise"):
        table = make_precomputed_foundation_feature_table([[tiny, 1.0]])
        aligned = align_precomputed_foundation_features(table, [0])

    assert table.features.dtype == np.float64
    assert aligned.dtype == np.float64
    assert table.features[0, 0] == tiny
    assert aligned[0, 0] == tiny
    assert table.features[0, 0] != 0.0
    assert aligned[0, 0] != 0.0


def test_precomputed_foundation_keeps_float32_for_ordinary_values() -> None:
    table = make_precomputed_foundation_feature_table([[1.0, 2.0], [3.0, 4.0]])
    aligned = align_precomputed_foundation_features(table, [1, 0])

    assert table.features.dtype == np.float32
    assert aligned.dtype == np.float32
    np.testing.assert_array_equal(aligned, [[3.0, 4.0], [1.0, 2.0]])
