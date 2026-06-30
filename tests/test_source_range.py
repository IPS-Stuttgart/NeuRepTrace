from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_range import (
    SOURCE_RANGE_CATEGORY,
    apply_source_range_clip,
    source_feature_range,
    source_range_clip,
)


def test_source_feature_range_uses_source_rows_only() -> None:
    source = np.asarray([[0.0, 10.0], [2.0, 12.0], [1.0, 11.0]], dtype=float)

    lower, upper = source_feature_range(source)

    assert SOURCE_RANGE_CATEGORY == "1_strict_source_only"
    assert np.allclose(lower, np.asarray([0.0, 10.0]))
    assert np.allclose(upper, np.asarray([2.0, 12.0]))


def test_source_range_clip_clips_test_rows_with_source_bounds() -> None:
    source = np.asarray([[0.0, 10.0], [2.0, 12.0]], dtype=float)
    test = np.asarray([[-5.0, 11.0], [5.0, 20.0]], dtype=float)

    train, test_out, lower, upper, train_mask, test_mask, metadata = source_range_clip(source_features=source, test_features=test)

    assert np.allclose(train, source)
    assert np.allclose(test_out, np.asarray([[0.0, 11.0], [2.0, 12.0]]))
    assert np.allclose(lower, np.asarray([0.0, 10.0]))
    assert np.allclose(upper, np.asarray([2.0, 12.0]))
    assert not np.any(train_mask)
    assert test_mask.tolist() == [[True, False], [True, True]]
    assert metadata["source_range_protocol_category"] == SOURCE_RANGE_CATEGORY
    assert metadata["source_range_uses_test_features_for_fitting"] is False
    assert metadata["source_range_uses_test_labels"] is False


def test_apply_source_range_clip_validates_width() -> None:
    with pytest.raises(ValueError, match="width"):
        apply_source_range_clip([[0.0, 1.0]], lower=[0.0], upper=[1.0])


def test_apply_source_range_clip_rejects_nonfinite_bounds() -> None:
    with pytest.raises(ValueError, match="finite"):
        apply_source_range_clip([[0.0, 1.0]], lower=[0.0, np.nan], upper=[1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        apply_source_range_clip([[0.0, 1.0]], lower=[0.0, 0.0], upper=[1.0, np.inf])


def test_apply_source_range_clip_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError, match="lower bounds.*upper"):
        apply_source_range_clip([[0.0, 1.0]], lower=[1.0, 0.0], upper=[0.0, 2.0])


def test_source_feature_range_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        source_feature_range([[0.0], [float("nan")]])


def test_heldout_arguments_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        source_feature_range([[0.0], [1.0]], heldout_features=[[0.5]])  # type: ignore[call-arg]
