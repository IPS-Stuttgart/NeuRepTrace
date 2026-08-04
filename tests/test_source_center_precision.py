from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_center import (
    SourceCenterConfig,
    SourceCenterMap,
    apply_source_center_transform,
    fit_source_center_map,
    fit_source_center_transform,
)


def test_direct_source_center_config_normalizes_aliases() -> None:
    config = SourceCenterConfig(center="average")

    assert config.center == "mean"
    center_map = fit_source_center_map([[0.0], [2.0]], config=config)
    assert center_map.center.tolist() == [1.0]


def test_source_center_mean_does_not_overflow_finite_source_values() -> None:
    source = np.asarray([[1e308, -1e308], [1e308, -1e308]])

    result = fit_source_center_transform(
        source_features=source,
        test_features=source,
        config={"center": "mean"},
    )

    np.testing.assert_allclose(result.center_map.center, [1e308, -1e308])
    np.testing.assert_array_equal(result.train_features, np.zeros_like(source))
    np.testing.assert_array_equal(result.test_features, np.zeros_like(source))


def test_source_center_preserves_float64_when_float32_would_corrupt() -> None:
    features = np.asarray([[1e300, 1e-300], [-1e300, -1e-300]])

    result = fit_source_center_transform(
        source_features=features,
        test_features=features,
        config={"center": "zero"},
    )

    assert result.train_features.dtype == np.float64
    assert result.test_features.dtype == np.float64
    np.testing.assert_array_equal(result.train_features, features)
    np.testing.assert_array_equal(result.test_features, features)


def test_source_center_rejects_unrepresentable_centered_differences() -> None:
    center_map = SourceCenterMap(
        center=np.asarray([1e308]),
        center_mode="mean",
        n_source_rows=2,
    )

    with pytest.raises(ValueError, match="exceed the finite floating-point range"):
        apply_source_center_transform([[-1e308]], center_map)
