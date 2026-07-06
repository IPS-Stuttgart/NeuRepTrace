from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_center import (
    SOURCE_CENTER_CATEGORY,
    apply_source_center_transform,
    fit_source_center_map,
    fit_source_center_transform,
    normalize_center_mode,
)


def test_source_center_uses_source_mean_only() -> None:
    source = np.asarray([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]], dtype=float)
    rows = np.asarray([[1.0, 19.0], [3.0, 21.0]], dtype=float)

    result = fit_source_center_transform(source_features=source, test_features=rows, config={"center": "mean"})

    assert result.center_map.center.tolist() == [2.0, 20.0]
    assert np.allclose(result.train_features.mean(axis=0), 0.0)
    assert result.test_features.tolist() == [[-1.0, -1.0], [1.0, 1.0]]
    assert result.metadata["source_center_protocol_category"] == SOURCE_CENTER_CATEGORY
    assert result.metadata["source_center_uses_test_features_for_fitting"] is False
    assert result.metadata["source_center_uses_labels"] is False
    assert result.metadata["source_center_valid_for_strict_source_only"] is True


def test_source_center_median_and_zero_modes() -> None:
    median_map = fit_source_center_map([[0.0], [1.0], [100.0]], config={"center": "median"})
    zero_map = fit_source_center_map([[0.0], [1.0], [100.0]], config={"center": "zero"})

    assert median_map.center.tolist() == [1.0]
    assert zero_map.center.tolist() == [0.0]
    assert apply_source_center_transform([[2.0]], median_map).ravel().tolist() == [1.0]


def test_source_center_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        fit_source_center_transform(source_features=[[0.0, 1.0]], test_features=[[0.0]])


def test_source_center_aliases_and_validation() -> None:
    assert normalize_center_mode("avg") == "mean"
    assert normalize_center_mode("med") == "median"
    assert normalize_center_mode("none") == "zero"

    with pytest.raises(ValueError, match="center mode"):
        normalize_center_mode("bad")
