from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_threshold import (
    SOURCE_THRESHOLD_CATEGORY,
    SourceThresholdConfig,
    apply_source_threshold_transform,
    fit_source_threshold_map,
    fit_source_threshold_transform,
    normalize_output_mode,
    normalize_threshold_mode,
    source_threshold_config,
)


def test_source_threshold_uses_source_thresholds_only() -> None:
    source = np.asarray([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]], dtype=float)
    rows = np.asarray([[1.0, 19.0], [3.0, 21.0]], dtype=float)

    result = fit_source_threshold_transform(source_features=source, test_features=rows, config={"threshold_mode": "median"})

    assert result.threshold_map.thresholds.tolist() == [2.0, 20.0]
    assert result.test_features.tolist() == [[0.0, 0.0], [1.0, 1.0]]
    assert result.metadata["source_threshold_protocol_category"] == SOURCE_THRESHOLD_CATEGORY
    assert result.metadata["source_threshold_uses_test_features_for_fitting"] is False
    assert result.metadata["source_threshold_uses_labels"] is False
    assert result.metadata["source_threshold_valid_for_strict_source_only"] is True


def test_source_threshold_quantile_and_signed_output() -> None:
    source = np.asarray([[0.0], [10.0], [20.0], [30.0]], dtype=float)
    threshold_map = fit_source_threshold_map(source, config={"threshold_mode": "quantile", "quantile": 0.75, "output": "signed"})

    out = apply_source_threshold_transform([[0.0], [25.0], [30.0]], threshold_map)

    assert np.isclose(threshold_map.thresholds[0], 22.5)
    assert out.ravel().tolist() == [-1.0, 1.0, 1.0]


def test_source_threshold_custom_binary_values() -> None:
    result = fit_source_threshold_transform(
        source_features=[[-1.0], [1.0]],
        test_features=[[-2.0], [2.0]],
        config={"threshold_mode": "zero", "positive_value": 2.0, "negative_value": -3.0},
    )

    assert result.test_features.ravel().tolist() == [-3.0, 2.0]


def test_source_threshold_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        fit_source_threshold_transform(source_features=[[0.0, 1.0]], test_features=[[0.0]])


def test_source_threshold_aliases_and_validation() -> None:
    assert normalize_threshold_mode("avg") == "mean"
    assert normalize_output_mode("pm1") == "signed"
    assert source_threshold_config(quantile="0.25").quantile == 0.25

    with pytest.raises(ValueError, match="threshold_mode"):
        normalize_threshold_mode("bad")

    with pytest.raises(ValueError, match="output"):
        normalize_output_mode("bad")

    with pytest.raises(ValueError, match="quantile"):
        source_threshold_config(quantile=1.5)


def test_source_threshold_rejects_boolean_numeric_config() -> None:
    with pytest.raises(ValueError, match="quantile"):
        source_threshold_config(quantile=True)

    with pytest.raises(ValueError, match="positive_value"):
        source_threshold_config(positive_value=np.asarray(True))

    with pytest.raises(ValueError, match="negative_value"):
        source_threshold_config(negative_value=np.asarray([0.0]))


def test_source_threshold_direct_config_is_normalized_and_validated() -> None:
    cfg = SourceThresholdConfig(
        threshold_mode="avg",
        quantile=np.asarray("0.25"),
        output="pm1",
        positive_value=np.asarray(2.0),
        negative_value="-3",
    )

    assert cfg.threshold_mode == "mean"
    assert cfg.quantile == 0.25
    assert cfg.output == "signed"
    assert cfg.positive_value == 2.0
    assert cfg.negative_value == -3.0

    threshold_map = fit_source_threshold_map([[0.0], [4.0]], config=cfg)
    out = apply_source_threshold_transform([[0.0], [2.0], [4.0]], threshold_map)
    assert out.ravel().tolist() == [-2.0, 2.0, 2.0]

    with pytest.raises(ValueError, match="threshold_mode"):
        SourceThresholdConfig(threshold_mode="bad")

    with pytest.raises(ValueError, match="positive_value"):
        SourceThresholdConfig(positive_value=True)
