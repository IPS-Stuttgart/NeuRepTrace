from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_feature_select import (
    SOURCE_FEATURE_SELECT_CATEGORY,
    select_source_variance_features,
    source_variance_feature_indices,
)


def test_select_source_variance_features_uses_source_scores_only() -> None:
    source = np.asarray(
        [
            [0.0, 0.0, 1.0, 5.0],
            [0.0, 1.0, 1.0, 7.0],
            [0.0, 2.0, 1.0, 9.0],
        ],
        dtype=float,
    )
    test = np.asarray([[10.0, 100.0, 1.0, 6.0]], dtype=float)

    result = select_source_variance_features(source_features=source, test_features=test, k=2)

    assert result.selected_indices.tolist() == [1, 3]
    assert result.train_features.shape == (3, 2)
    assert result.test_features.shape == (1, 2)
    assert np.allclose(result.test_features, np.asarray([[100.0, 6.0]]))
    assert result.metadata["source_feature_select_protocol_category"] == SOURCE_FEATURE_SELECT_CATEGORY
    assert result.metadata["source_feature_select_uses_source_features"] is True
    assert result.metadata["source_feature_select_uses_test_features_for_fitting"] is False
    assert result.metadata["source_feature_select_uses_test_labels"] is False
    assert result.metadata["source_feature_select_valid_for_strict_source_only"] is True


def test_select_source_variance_features_accepts_integral_string_k_in_metadata() -> None:
    source = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 3.0],
            [0.0, 2.0, 6.0],
        ],
        dtype=float,
    )
    test = np.asarray([[10.0, 11.0, 12.0]], dtype=float)

    result = select_source_variance_features(source_features=source, test_features=test, k="2.0")

    assert result.selected_indices.tolist() == [1, 2]
    assert result.metadata["source_feature_select_k"] == 2


def test_source_variance_feature_indices_respects_min_variance() -> None:
    selected = source_variance_feature_indices(scores=[0.0, 0.5, 2.0, 1.0], min_variance=0.75)

    assert selected.tolist() == [2, 3]


def test_source_variance_feature_indices_falls_back_to_best_feature() -> None:
    selected = source_variance_feature_indices(scores=[0.0, 0.5, 0.2], min_variance=2.0)

    assert selected.tolist() == [1]


def test_source_variance_feature_indices_validates_inputs() -> None:
    with pytest.raises(ValueError, match="scores"):
        source_variance_feature_indices(scores=[0.0, -1.0])
    with pytest.raises(ValueError, match="k"):
        source_variance_feature_indices(scores=[0.0, 1.0], k=0)
    with pytest.raises(ValueError, match="min_variance"):
        source_variance_feature_indices(scores=[0.0, 1.0], min_variance=-1.0)


def test_select_source_variance_features_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        select_source_variance_features(source_features=[[0.0, 1.0]], test_features=[[0.0]])


def test_heldout_arguments_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        select_source_variance_features(
            source_features=[[0.0], [1.0]],
            test_features=[[0.5]],
            heldout_labels=[0],  # type: ignore[call-arg]
        )
