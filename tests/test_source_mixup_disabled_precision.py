from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_mixup import augment_source_with_mixup


def test_disabled_source_mixup_preserves_finite_float64_range() -> None:
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
        result = augment_source_with_mixup(features, labels)

    assert result.features.dtype == np.float64
    assert np.all(np.isfinite(result.features))
    np.testing.assert_array_equal(result.features, features)


def test_disabled_source_mixup_keeps_safe_outputs_compact() -> None:
    features = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    labels = np.asarray([0, 1])

    result = augment_source_with_mixup(features, labels)

    assert result.features.dtype == np.float32
    np.testing.assert_array_equal(result.features, features.astype(np.float32))
