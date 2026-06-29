from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_scale import (
    SOURCE_SCALE_CATEGORY,
    apply_source_feature_scale,
    fit_source_feature_scale,
    fit_source_feature_scale_stats,
    normalize_source_scale_method,
    source_feature_scale_config,
)


def test_standard_scale_fits_source_only_and_transforms_test_rows() -> None:
    source = np.asarray([[0.0, 10.0], [2.0, 12.0], [4.0, 14.0]], dtype=float)
    test = np.asarray([[6.0, 16.0]], dtype=float)

    result = fit_source_feature_scale(source_features=source, test_features=test, config={"method": "standard"})

    assert np.allclose(np.mean(result.train_features, axis=0), 0.0)
    assert result.test_features.shape == test.shape
    assert result.metadata["source_feature_scale_protocol_category"] == SOURCE_SCALE_CATEGORY
    assert result.metadata["source_feature_scale_uses_source_features"] is True
    assert result.metadata["source_feature_scale_uses_test_features_for_fitting"] is False
    assert result.metadata["source_feature_scale_uses_test_labels"] is False
    assert result.metadata["source_feature_scale_valid_for_strict_source_only"] is True


def test_robust_scale_uses_median_and_iqr() -> None:
    source = np.asarray([[0.0], [1.0], [2.0], [100.0]], dtype=float)
    stats = fit_source_feature_scale_stats(source, config={"method": "robust"})

    assert np.isclose(stats.offset[0], 1.5)
    assert stats.scale[0] > 0.0
    transformed = apply_source_feature_scale(source, stats)
    assert transformed.shape == source.shape


def test_minmax_scale_maps_source_range_to_unit_interval() -> None:
    source = np.asarray([[1.0], [3.0], [5.0]], dtype=float)
    result = fit_source_feature_scale(source_features=source, test_features=[[7.0]], config={"method": "minmax"})

    assert np.allclose(result.train_features.ravel(), np.asarray([0.0, 0.5, 1.0]))
    assert np.allclose(result.test_features.ravel(), np.asarray([1.5]))


def test_scaling_can_disable_center_or_scale() -> None:
    source = np.asarray([[1.0], [3.0]], dtype=float)
    no_center = fit_source_feature_scale(source_features=source, test_features=[[5.0]], config={"center": False, "scale": True})
    no_scale = fit_source_feature_scale(source_features=source, test_features=[[5.0]], config={"center": True, "scale": False})

    assert not np.allclose(no_center.train_features, no_scale.train_features)
    assert np.allclose(no_scale.train_features.ravel(), np.asarray([-1.0, 1.0]))


def test_aliases_and_validation() -> None:
    assert normalize_source_scale_method("z-score") == "standard"
    assert normalize_source_scale_method("median-iqr") == "robust"
    assert normalize_source_scale_method("min-max") == "minmax"
    assert source_feature_scale_config(center="false").center is False

    with pytest.raises(ValueError, match="source scale method"):
        normalize_source_scale_method("bad")

    with pytest.raises(ValueError, match="epsilon"):
        source_feature_scale_config(epsilon=0.0)


@pytest.mark.parametrize("value", [True, np.bool_(True), [], {"epsilon": 1}, np.asarray(1e-8), np.asarray([1e-8])])
def test_source_scale_rejects_non_numeric_epsilon_values(value: object) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        source_feature_scale_config(epsilon=value)  # type: ignore[arg-type]


def test_source_scale_rejects_width_mismatch_and_extra_labels() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        fit_source_feature_scale(source_features=[[0.0, 1.0]], test_features=[[0.0]])
    with pytest.raises(TypeError):
        fit_source_feature_scale(source_features=[[0.0], [1.0]], test_features=[[0.5]], test_labels=[0])  # type: ignore[call-arg]
