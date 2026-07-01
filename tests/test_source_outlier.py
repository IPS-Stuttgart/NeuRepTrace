from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_outlier import (
    SOURCE_OUTLIER_CATEGORY,
    SourceOutlierConfig,
    compute_source_outlier_weights,
    normalize_threshold_mode,
    normalize_weight_mode,
    source_outlier_config,
)


def test_source_outlier_weights_downweight_far_class_member() -> None:
    features = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [4.0, 4.0],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
            [6.0, 6.0],
        ],
        dtype=float,
    )
    labels = np.asarray(["a", "a", "a", "a", "b", "b", "b", "b"], dtype=object)

    result = compute_source_outlier_weights(
        features,
        labels,
        config={"threshold_mode": "quantile", "quantile": 0.75, "weight_mode": "binary"},
    )

    assert result.distances.shape == (8,)
    assert result.sample_weights.shape == (8,)
    assert result.inlier_mask.shape == (8,)
    assert result.metadata["source_outlier_protocol_category"] == SOURCE_OUTLIER_CATEGORY
    assert result.metadata["source_outlier_uses_heldout_features"] is False
    assert result.metadata["source_outlier_uses_heldout_labels"] is False
    assert result.metadata["source_outlier_valid_for_strict_source_only"] is True
    assert result.sample_weights[3] == 0.0
    assert result.sample_weights[7] == 0.0
    assert result.sample_weights[0] == 1.0


def test_source_outlier_soft_weights_are_between_zero_and_one() -> None:
    features = np.asarray([[-1.0], [-0.5], [0.0], [8.0], [10.0], [10.5], [11.0], [3.0]], dtype=float)
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=object)

    result = compute_source_outlier_weights(
        features,
        labels,
        config={"threshold_mode": "mad", "mad_multiplier": 1.0, "weight_mode": "soft", "temperature": 0.5},
    )

    assert np.all(result.sample_weights > 0.0)
    assert np.all(result.sample_weights <= 1.0)
    assert np.any(result.sample_weights < 1.0)
    assert result.metadata["source_outlier_threshold_mode"] == "mad"
    assert result.metadata["source_outlier_weight_mode"] == "soft"


def test_source_outlier_linear_weights_ramp_after_threshold() -> None:
    features = np.asarray([[0.0], [1.0], [2.0], [8.0], [10.0], [11.0]], dtype=float)
    labels = np.asarray(["x", "x", "x", "y", "y", "y"], dtype=object)

    result = compute_source_outlier_weights(
        features,
        labels,
        config={"threshold_mode": "quantile", "quantile": 0.5, "weight_mode": "linear", "use_diagonal_scale": False},
    )

    assert np.all(result.sample_weights >= 0.0)
    assert np.all(result.sample_weights <= 1.0)
    assert np.any((result.sample_weights > 0.0) & (result.sample_weights < 1.0))


def test_source_outlier_aliases_and_validation() -> None:
    assert normalize_threshold_mode("percentile") == "quantile"
    assert normalize_threshold_mode("median-absolute-deviation") == "mad"
    assert normalize_weight_mode("hard") == "binary"
    assert normalize_weight_mode("exponential") == "soft"
    cfg = source_outlier_config(quantile="0.8", temperature="2.0")
    assert cfg.quantile == 0.8
    assert cfg.temperature == 2.0

    with pytest.raises(ValueError, match="threshold_mode"):
        normalize_threshold_mode("bad")

    with pytest.raises(ValueError, match="weight_mode"):
        normalize_weight_mode("bad")

    with pytest.raises(ValueError, match="quantile"):
        source_outlier_config(quantile=1.5)


def test_source_outlier_boolean_config_parses_string_values() -> None:
    cfg = source_outlier_config(use_diagonal_scale="false")
    assert cfg.use_diagonal_scale is False
    cfg = source_outlier_config(use_diagonal_scale="ON")
    assert cfg.use_diagonal_scale is True


def test_source_outlier_revalidates_dataclass_config_instances() -> None:
    features = np.asarray([[0.0], [1.0], [2.0], [8.0], [10.0], [11.0]], dtype=float)
    labels = np.asarray(["x", "x", "x", "y", "y", "y"], dtype=object)
    config = SourceOutlierConfig(use_diagonal_scale="false")  # type: ignore[arg-type]

    result = compute_source_outlier_weights(features, labels, config=config)

    assert np.all(result.feature_scale == 1.0)


def test_source_outlier_rejects_boolean_numeric_config_values() -> None:
    for name in ("quantile", "mad_multiplier", "temperature", "epsilon"):
        with pytest.raises(ValueError, match=name):
            source_outlier_config(**{name: True})


def test_source_outlier_requires_matching_label_rows() -> None:
    with pytest.raises(ValueError, match="source_labels"):
        compute_source_outlier_weights([[0.0], [1.0]], [0])
