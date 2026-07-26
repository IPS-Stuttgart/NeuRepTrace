from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_scale import fit_source_feature_scale


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("standard", [-np.sqrt(3.0) / 2.0, -np.sqrt(3.0) / 2.0, np.sqrt(3.0) / 2.0, np.sqrt(3.0) / 2.0]),
        ("robust", [-0.6745, -0.6745, 0.6745, 0.6745]),
        ("minmax", [0.0, 0.0, 1.0, 1.0]),
    ],
)
def test_source_scale_handles_full_float64_range(method: str, expected: list[float]) -> None:
    maximum = np.finfo(float).max
    source = np.asarray([[-maximum], [-maximum], [maximum], [maximum]], dtype=float)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = fit_source_feature_scale(
            source_features=source,
            test_features=[[0.0]],
            config={"method": method},
        )

    assert np.all(np.isfinite(result.stats.offset))
    assert np.all(np.isfinite(result.stats.scale))
    assert np.all(np.isfinite(result.train_features))
    assert np.all(np.isfinite(result.test_features))
    np.testing.assert_allclose(result.train_features.ravel(), expected, rtol=1e-6, atol=1e-7)
