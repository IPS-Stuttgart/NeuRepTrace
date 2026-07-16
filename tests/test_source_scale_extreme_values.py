from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_scale import fit_source_feature_scale


def test_standard_scale_avoids_overflow_for_extreme_finite_values() -> None:
    source = np.asarray([[1e308, 1e308], [1e308, -1e308]], dtype=float)

    with np.errstate(over="raise", invalid="raise"):
        result = fit_source_feature_scale(
            source_features=source,
            test_features=source,
            config={"method": "standard"},
        )

    np.testing.assert_allclose(result.stats.offset, [1e308, 0.0])
    np.testing.assert_allclose(result.stats.scale, [1e-8, np.sqrt(2.0) * 1e308], rtol=1e-15)
    assert np.all(np.isfinite(result.train_features))
    assert np.all(np.isfinite(result.test_features))
    np.testing.assert_allclose(result.train_features[:, 0], [0.0, 0.0])
    np.testing.assert_allclose(
        result.train_features[:, 1],
        [1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)],
        rtol=1e-6,
    )
