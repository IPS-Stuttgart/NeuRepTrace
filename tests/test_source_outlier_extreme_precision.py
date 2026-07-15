from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_outlier import compute_source_outlier_weights


def test_source_outlier_preserves_large_finite_statistics() -> None:
    features = np.asarray(
        [
            [1e308, 0.0],
            [1e308, 1.0],
            [-1e308, 0.0],
            [-1e308, 1.0],
        ],
        dtype=float,
    )
    labels = np.asarray(["a", "a", "b", "b"], dtype=object)

    with np.errstate(over="raise", invalid="raise"):
        result = compute_source_outlier_weights(
            features,
            labels,
            config={"threshold_mode": "quantile", "quantile": 1.0},
        )

    assert result.centroids.dtype == np.float64
    assert result.feature_scale.dtype == np.float64
    assert np.all(np.isfinite(result.centroids))
    assert np.all(np.isfinite(result.feature_scale))
    assert np.all(np.isfinite(result.distances))
    assert np.all(np.isfinite(result.sample_weights))
    np.testing.assert_allclose(result.centroids[:, 0], [1e308, -1e308], rtol=1e-15, atol=0.0)
    np.testing.assert_allclose(result.feature_scale[0], np.sqrt(4.0 / 3.0) * 1e308, rtol=1e-15, atol=0.0)
    np.testing.assert_allclose(result.distances, np.full(4, np.sqrt(3.0) / 2.0), rtol=1e-6, atol=0.0)


def test_source_outlier_preserves_large_finite_unscaled_distances() -> None:
    features = np.asarray(
        [
            [1e200, 1e200],
            [-1e200, -1e200],
            [1e200, -1e200],
            [-1e200, 1e200],
        ],
        dtype=float,
    )
    labels = np.asarray(["a", "a", "b", "b"], dtype=object)

    with np.errstate(over="raise", invalid="raise"):
        result = compute_source_outlier_weights(
            features,
            labels,
            config={
                "threshold_mode": "quantile",
                "quantile": 1.0,
                "use_diagonal_scale": False,
            },
        )

    assert result.distances.dtype == np.float64
    assert np.all(np.isfinite(result.distances))
    np.testing.assert_allclose(result.distances, np.full(4, np.sqrt(2.0) * 1e200), rtol=1e-12, atol=0.0)


def test_source_outlier_keeps_float32_for_ordinary_values() -> None:
    result = compute_source_outlier_weights(
        [[0.0], [1.0], [10.0], [11.0]],
        ["a", "a", "b", "b"],
    )

    assert result.distances.dtype == np.float32
    assert result.sample_weights.dtype == np.float32
    assert result.centroids.dtype == np.float32
    assert result.feature_scale.dtype == np.float32
