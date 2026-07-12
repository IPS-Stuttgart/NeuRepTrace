from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_zca import fit_source_zca_transform


def test_source_zca_preserves_extreme_finite_reference_precision() -> None:
    with np.errstate(over="raise", under="raise", invalid="raise"):
        result = fit_source_zca_transform(
            source_features=[[0.0], [0.0]],
            test_features=[[1.0]],
            config={"regularization": 1e-100, "recolor": True},
        )

    assert result.reference.whitening.dtype == np.float64
    assert result.reference.coloring.dtype == np.float64
    assert np.all(np.isfinite(result.reference.whitening))
    assert np.all(np.isfinite(result.reference.coloring))
    assert np.all(np.isfinite(result.train_features))
    assert np.all(np.isfinite(result.test_features))
    np.testing.assert_allclose(result.test_features, [[1.0]], rtol=1e-12, atol=0.0)


def test_source_zca_keeps_float32_for_ordinary_values() -> None:
    result = fit_source_zca_transform(
        source_features=[[0.0], [1.0]],
        test_features=[[0.5]],
    )

    assert result.reference.whitening.dtype == np.float32
    assert result.reference.coloring.dtype == np.float32
    assert result.train_features.dtype == np.float32
    assert result.test_features.dtype == np.float32
