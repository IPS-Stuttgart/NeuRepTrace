from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_robust_scaler import (
    SOURCE_ROBUST_SCALER_CATEGORY,
    apply_source_robust_scaler,
    fit_source_robust_scaler,
    fit_source_robust_scaler_stats,
    normalize_center_mode,
    normalize_scale_mode,
    source_robust_scaler_config,
)


def test_source_robust_scaler_uses_source_statistics_only() -> None:
    source = np.asarray([[0.0, 10.0], [1.0, 12.0], [2.0, 14.0]], dtype=float)
    test = np.asarray([[100.0, 100.0], [1.0, 12.0]], dtype=float)

    result = fit_source_robust_scaler(
        source_features=source,
        test_features=test,
        config={"center": "median", "scale": "iqr", "lower_quantile": 0.25, "upper_quantile": 0.75},
    )

    assert np.allclose(result.stats.location, np.asarray([1.0, 12.0]))
    assert np.allclose(result.stats.scale, np.asarray([1.0, 2.0]))
    assert np.allclose(result.train_features[1], np.asarray([0.0, 0.0]))
    assert np.allclose(result.test_features[1], np.asarray([0.0, 0.0]))
    assert result.metadata["source_robust_scaler_protocol_category"] == SOURCE_ROBUST_SCALER_CATEGORY
    assert result.metadata["source_robust_scaler_uses_test_features_for_fitting"] is False
    assert result.metadata["source_robust_scaler_valid_for_strict_source_only"] is True


def test_source_robust_scaler_supports_mad_and_mean_modes() -> None:
    source = np.asarray([[0.0], [1.0], [2.0]], dtype=float)
    stats = fit_source_robust_scaler_stats(source, config={"center": "mean", "scale": "mad"})

    assert np.allclose(stats.location, np.asarray([1.0]))
    assert np.allclose(stats.scale, np.asarray([1.4826]))


def test_apply_source_robust_scaler_rejects_width_mismatch() -> None:
    stats = fit_source_robust_scaler_stats([[0.0, 1.0], [1.0, 2.0]])

    with pytest.raises(ValueError, match="features width"):
        apply_source_robust_scaler([[0.0]], stats=stats)


def test_source_robust_scaler_config_aliases_and_validation() -> None:
    assert normalize_center_mode("avg") == "mean"
    assert normalize_center_mode("off") == "none"
    assert normalize_scale_mode("median-absolute-deviation") == "mad"
    assert normalize_scale_mode("sd") == "std"

    cfg = source_robust_scaler_config(lower_quantile="0.1", upper_quantile="0.9", epsilon="1e-6")
    assert cfg.lower_quantile == 0.1
    assert cfg.upper_quantile == 0.9
    assert np.isclose(cfg.epsilon, 1e-6)

    with pytest.raises(ValueError, match="lower_quantile"):
        source_robust_scaler_config(lower_quantile=0.9, upper_quantile=0.1)

    with pytest.raises(ValueError, match="center"):
        normalize_center_mode("bad")

    with pytest.raises(ValueError, match="scale"):
        normalize_scale_mode("bad")


def test_source_robust_scaler_rejects_feature_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        fit_source_robust_scaler(source_features=[[0.0, 1.0]], test_features=[[0.0]])
