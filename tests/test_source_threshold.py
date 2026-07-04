from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_threshold import (
    SOURCE_THRESHOLD_CATEGORY,
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
