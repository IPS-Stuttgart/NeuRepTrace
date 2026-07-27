from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_asinh import fit_source_asinh_transform


def test_source_asinh_remains_finite_for_extreme_finite_ratio() -> None:
    max_float = np.finfo(float).max
    min_positive = np.nextafter(0.0, 1.0)

    result = fit_source_asinh_transform(
        source_features=[[0.0], [0.0]],
        test_features=[[-max_float], [max_float]],
        config={"scale_mode": "mad", "epsilon": min_positive},
    )

    expected_magnitude = np.log(max_float) - np.log(min_positive) + np.log(2.0)
    assert np.all(np.isfinite(result.test_features))
    assert np.allclose(
        result.test_features.ravel(),
        [-expected_magnitude, expected_magnitude],
        rtol=1e-6,
        atol=0.0,
    )


def test_source_asinh_compaction_preserves_subnormal_nonzero_outputs() -> None:
    min_positive = np.nextafter(0.0, 1.0)

    result = fit_source_asinh_transform(
        source_features=[[min_positive]],
        test_features=[[-min_positive]],
        config={"scale_mode": "unit"},
    )

    assert result.train_features[0, 0] != 0.0
    assert result.test_features[0, 0] != 0.0
    assert np.all(np.isfinite(result.train_features))
    assert np.all(np.isfinite(result.test_features))
