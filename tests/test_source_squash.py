from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_squash import (
    SOURCE_SQUASH_CATEGORY,
    apply_source_squash_transform,
    fit_source_squash_map,
    fit_source_squash_transform,
    normalize_scale_mode,
    source_squash_config,
)


def test_source_squash_uses_source_scales_only() -> None:
    source = np.asarray([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]], dtype=float)
    rows = np.asarray([[1.0, 2.0], [10.0, 20.0]], dtype=float)

    result = fit_source_squash_transform(source_features=source, test_features=rows, config={"scale_mode": "unit"})

    expected = rows / (1.0 + np.abs(rows))
    assert np.allclose(result.test_features, expected)
    assert result.metadata["source_squash_protocol_category"] == SOURCE_SQUASH_CATEGORY
    assert result.metadata["source_squash_uses_test_features_for_fitting"] is False
    assert result.metadata["source_squash_uses_labels"] is False
    assert result.metadata["source_squash_valid_for_strict_source_only"] is True


def test_source_squash_mad_scale_matches_manual_transform() -> None:
    source = np.asarray([[0.0], [1.0], [2.0]], dtype=float)
    transform_map = fit_source_squash_map(source, config={"scale_mode": "mad"})

    out = apply_source_squash_transform([[1.0], [2.0]], transform_map)
    scaled = np.asarray([1.0, 2.0]) / 1.4826

    assert np.isclose(transform_map.scale[0], 1.4826)
    assert np.allclose(out.ravel(), scaled / (1.0 + np.abs(scaled)))


def test_source_squash_iqr_multiplier_changes_scale() -> None:
    source = np.asarray([[0.0], [10.0], [20.0], [30.0]], dtype=float)
    first = fit_source_squash_map(source, config={"scale_mode": "iqr", "multiplier": 1.0})
    second = fit_source_squash_map(source, config={"scale_mode": "iqr", "multiplier": 2.0})

    assert np.allclose(second.scale, 2.0 * first.scale)


def test_source_squash_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        fit_source_squash_transform(source_features=[[0.0, 1.0]], test_features=[[0.0]])


def test_source_squash_aliases_and_validation() -> None:
    assert normalize_scale_mode("robust") == "mad"
    assert normalize_scale_mode("interquartile") == "iqr"
    assert source_squash_config(multiplier="2.5").multiplier == 2.5

    with pytest.raises(ValueError, match="scale_mode"):
        normalize_scale_mode("bad")

    with pytest.raises(ValueError, match="multiplier"):
        source_squash_config(multiplier=0.0)
