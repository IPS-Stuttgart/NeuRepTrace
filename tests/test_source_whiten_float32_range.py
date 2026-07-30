from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_whiten import fit_source_whiten


def test_source_whiten_preserves_finite_values_outside_float32_range() -> None:
    result = fit_source_whiten(
        source_features=[[0.0], [0.0]],
        test_features=[[1.0]],
        config={"method": "pca", "regularization": 1e-300},
    )

    assert np.all(np.isfinite(result.transform.whitening))
    assert np.all(np.isfinite(result.train_features))
    assert np.all(np.isfinite(result.test_features))
    assert result.transform.whitening.dtype == np.float64
    assert result.test_features.dtype == np.float64
    assert result.test_features[0, 0] > np.finfo(np.float32).max
