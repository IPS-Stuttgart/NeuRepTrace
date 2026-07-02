from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_standardize import (
    SOURCE_STANDARDIZE_CATEGORY,
    fit_source_standardize_stats,
    fit_source_standardizer,
    normalize_location_mode,
    normalize_scale_mode,
    source_standardize_config,
    transform_with_source_standardizer,
)


def test_source_standardizer_uses_source_stats_only() -> None:
    source = np.asarray([[0.0], [1.0], [2.0]], dtype=float)
    test = np.asarray([[1.0], [10.0]], dtype=float)

    result = fit_source_standardizer(source_features=source, test_features=test)

    assert np.allclose(result.train_features.mean(axis=0), 0.0)
    assert result.test_features.shape == test.shape
    assert result.metadata["source_standardizer_protocol_category"] == SOURCE_STANDARDIZE_CATEGORY
    assert result.metadata["source_standardizer_uses_test_features_for_fitting"] is False
    assert result.metadata["source_standardizer_uses_test_labels"] is False
    assert result.metadata["source_standardizer_valid_for_strict_source_only"] is True


def test_robust_median_iqr_scaling() -> None:
    source = np.asarray([[0.0], [1.0], [2.0], [100.0]], dtype=float)
    stats = fit_source_standardize_stats(source, config={"location": "median", "scale": "iqr"})

    assert np.isclose(stats.location[0], 1.5)
    assert stats.scale[0] > 0.0

    transformed = transform_with_source_standardizer([[1.5]], stats, config={"location": "median", "scale": "iqr"})
    assert np.allclose(transformed, [[0.0]])


def test_clip_limits_standardized_output() -> None:
    source = np.asarray([[0.0], [1.0], [2.0]], dtype=float)
    test = np.asarray([[-100.0], [100.0]], dtype=float)

    result = fit_source_standardizer(source_features=source, test_features=test, config={"clip": 1.0})

    assert np.all(result.test_features <= 1.0)
    assert np.all(result.test_features >= -1.0)
    assert result.metadata["source_standardizer_clip"] == 1.0


def test_none_scale_and_zero_location_are_identity_for_values() -> None:
    source = np.asarray([[3.0], [3.0]], dtype=float)
    stats = fit_source_standardize_stats(source, config={"location": "zero", "scale": "none"})

    transformed = transform_with_source_standardizer([[3.0], [4.0]], stats, config={"location": "zero", "scale": "none"})

    assert np.allclose(transformed.ravel(), [3.0, 4.0])


def test_standardizer_aliases_and_validation() -> None:
    assert normalize_location_mode("med") == "median"
    assert normalize_location_mode("off") == "zero"
    assert normalize_scale_mode("median-abs-deviation") == "mad"
    assert normalize_scale_mode("unit") == "none"
    assert source_standardize_config(clip="2.0").clip == 2.0

    with pytest.raises(ValueError, match="location mode"):
        normalize_location_mode("bad")

    with pytest.raises(ValueError, match="scale mode"):
        normalize_scale_mode("bad")

    with pytest.raises(ValueError, match="same feature width"):
        fit_source_standardizer(source_features=[[0.0, 1.0]], test_features=[[0.0]])
