from __future__ import annotations

import numpy as np

from neureptrace.decoding import source_jitter
from neureptrace.decoding.source_jitter import augment_source_with_feature_jitter


def test_disabled_feature_jitter_preserves_finite_float64_range(monkeypatch) -> None:
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

    def unexpected_scale_computation(*args, **kwargs):
        raise AssertionError("disabled jitter must not compute unused feature scales")

    monkeypatch.setattr(source_jitter, "_scale_by_class", unexpected_scale_computation)
    with np.errstate(over="raise", invalid="raise"):
        result = augment_source_with_feature_jitter(features, labels)

    assert result.features.dtype == np.float64
    assert np.all(np.isfinite(result.features))
    np.testing.assert_array_equal(result.features, features)


def test_disabled_feature_jitter_keeps_safe_outputs_compact() -> None:
    features = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    labels = np.asarray([0, 1])

    result = augment_source_with_feature_jitter(features, labels)

    assert result.features.dtype == np.float32
    np.testing.assert_array_equal(result.features, features.astype(np.float32))


def test_disabled_feature_jitter_materializes_generator_rows() -> None:
    rows = ((value for value in row) for row in [[0.0, 1.0], [2.0, 3.0]])

    result = augment_source_with_feature_jitter(rows, [0, 1])

    np.testing.assert_array_equal(result.features, np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
