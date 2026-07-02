from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_clip import (
    SOURCE_CLIP_CATEGORY,
    apply_source_clip,
    fit_source_clip,
    fit_source_clip_bounds,
    normalize_center_mode,
    source_clip_config,
)


def test_source_clip_uses_source_bounds_only() -> None:
    source = np.asarray([[0.0], [1.0], [2.0], [100.0]], dtype=float)
    rows = np.asarray([[-10.0], [50.0], [200.0]], dtype=float)

    result = fit_source_clip(source_features=source, test_features=rows, config={"lower_quantile": 0.25, "upper_quantile": 0.75})

    assert result.metadata["source_clip_protocol_category"] == SOURCE_CLIP_CATEGORY
    assert result.metadata["source_clip_uses_test_features_for_fitting"] is False
    assert result.metadata["source_clip_uses_test_labels"] is False
    assert np.all(result.test_features >= result.bounds.lower)
    assert np.all(result.test_features <= result.bounds.upper)
    assert np.count_nonzero(result.test_clipped_mask) == 2


def test_symmetric_bounds_are_centered_on_median() -> None:
    source = np.asarray([[-2.0], [0.0], [2.0], [100.0]], dtype=float)

    bounds = fit_source_clip_bounds(source, config={"symmetric": True, "upper_quantile": 0.5, "center": "median"})

    assert np.isclose(bounds.center[0], 1.0)
    assert np.isclose(bounds.upper[0] - bounds.center[0], bounds.center[0] - bounds.lower[0])


def test_apply_source_clip_returns_changed_value_mask() -> None:
    bounds = fit_source_clip_bounds([[0.0], [1.0], [2.0]], config={"lower_quantile": 0.0, "upper_quantile": 1.0})

    clipped, mask = apply_source_clip([[-1.0], [1.0], [3.0]], bounds)

    assert clipped.ravel().tolist() == [0.0, 1.0, 2.0]
    assert mask.ravel().tolist() == [True, False, True]


def test_source_clip_aliases_and_validation() -> None:
    assert normalize_center_mode("med") == "median"
    assert normalize_center_mode("none") == "zero"
    assert source_clip_config(symmetric="true").symmetric is True

    with pytest.raises(ValueError, match="lower_quantile"):
        source_clip_config(lower_quantile=0.9, upper_quantile=0.1)

    with pytest.raises(ValueError, match="center"):
        normalize_center_mode("bad")
