from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_smote import augment_source_with_smote


def test_disabled_source_smote_preserves_finite_float64_range() -> None:
    limit = np.finfo(np.float64).max
    tiny = np.nextafter(0.0, 1.0)
    features = np.asarray(
        [
            [limit, tiny],
            [-limit, -tiny],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([0, 1])

    with np.errstate(over="ignore", under="ignore", invalid="raise"):
        result = augment_source_with_smote(features, labels)

    assert result.features.dtype == np.float64
    assert np.all(np.isfinite(result.features))
    np.testing.assert_array_equal(result.features, features)
