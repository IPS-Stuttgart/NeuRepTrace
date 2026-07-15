from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_outlier import compute_source_outlier_weights


def test_source_outlier_preserves_overflowing_finite_outputs() -> None:
    extreme = float(np.finfo(np.float32).max) * 4.0
    features = np.asarray([[0.0], [extreme], [10.0], [10.0]], dtype=float)
    labels = np.asarray(["a", "a", "b", "b"], dtype=object)

    with np.errstate(over="raise", under="raise", invalid="raise"):
        result = compute_source_outlier_weights(
            features,
            labels,
            config={"weight_mode": "binary", "use_diagonal_scale": False},
        )

    assert result.distances.dtype == np.float64
    assert result.centroids.dtype == np.float64
    assert np.all(np.isfinite(result.distances))
    assert np.all(np.isfinite(result.centroids))
    assert np.max(result.distances) > np.finfo(np.float32).max
    assert np.max(result.centroids) > np.finfo(np.float32).max


def test_source_outlier_preserves_underflowing_nonzero_weights() -> None:
    features = np.asarray(
        [[0.0], [120.0], [240.0], [1000.0], [1001.0], [1002.0]],
        dtype=float,
    )
    labels = np.asarray(["a", "a", "a", "b", "b", "b"], dtype=object)

    with np.errstate(over="raise", under="raise", invalid="raise"):
        result = compute_source_outlier_weights(
            features,
            labels,
            config={
                "threshold_mode": "quantile",
                "quantile": 0.0,
                "weight_mode": "soft",
                "temperature": 1.0,
                "use_diagonal_scale": False,
                "epsilon": 1.0,
            },
        )

    assert result.sample_weights.dtype == np.float64
    assert result.sample_weights[0] > 0.0
    assert result.sample_weights[2] > 0.0
    assert result.sample_weights[0] < np.finfo(np.float32).smallest_subnormal
    assert result.sample_weights[2] < np.finfo(np.float32).smallest_subnormal


def test_source_outlier_keeps_float32_for_ordinary_outputs() -> None:
    result = compute_source_outlier_weights(
        [[0.0], [1.0], [2.0], [8.0], [10.0], [11.0]],
        ["a", "a", "a", "b", "b", "b"],
        config={"weight_mode": "binary", "use_diagonal_scale": False},
    )

    assert result.distances.dtype == np.float32
    assert result.sample_weights.dtype == np.float32
    assert result.centroids.dtype == np.float32
    assert result.feature_scale.dtype == np.float32
