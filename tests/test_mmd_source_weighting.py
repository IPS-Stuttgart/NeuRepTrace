from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.mmd_source_weighting import (
    MMD_SOURCE_WEIGHTING_PROTOCOL_CATEGORY,
    mmd_source_group_scores,
    mmd_source_group_weights,
    resolve_mmd_gamma,
)


def test_mmd_source_weighting_upweights_distribution_near_unlabeled_target() -> None:
    source_features = {
        "near": np.asarray([[0.0, 0.0], [0.1, -0.1], [-0.1, 0.1]], dtype=float),
        "far": np.asarray([[4.0, 4.0], [4.2, 3.8], [3.8, 4.1]], dtype=float),
    }
    target_features = np.asarray([[0.0, 0.1], [0.2, -0.1], [-0.2, 0.0]], dtype=float)

    result = mmd_source_group_weights(
        source_features,
        target_features,
        gamma=1.0,
        temperature=0.25,
    )

    assert result.weights["near"] > result.weights["far"]
    assert result.mmd_squared["near"] < result.mmd_squared["far"]
    assert np.isclose(np.mean(list(result.weights.values())), 1.0)
    assert result.metadata["mmd_source_weighting_protocol_category"] == MMD_SOURCE_WEIGHTING_PROTOCOL_CATEGORY
    assert result.metadata["mmd_source_weighting_uses_unlabeled_target_data"] is True
    assert result.metadata["mmd_source_weighting_uses_target_labels"] is False
    assert result.metadata["mmd_source_weighting_valid_for_protocol_2_5"] is True


def test_mmd_top_k_keeps_only_lowest_mmd_source_before_mean_one_normalization() -> None:
    source_features = {
        "near": [[0.0], [0.1]],
        "middle": [[1.0], [1.1]],
        "far": [[5.0], [5.1]],
    }
    target_features = [[0.05], [0.0]]

    result = mmd_source_group_weights(
        source_features,
        target_features,
        gamma="scale",
        temperature=0.5,
        top_k=1,
    )

    assert result.weights["near"] == pytest.approx(3.0)
    assert result.weights["middle"] == pytest.approx(0.0)
    assert result.weights["far"] == pytest.approx(0.0)
    assert np.isclose(np.mean(list(result.weights.values())), 1.0)


def test_mmd_blend_can_make_weights_conservative() -> None:
    source_features = {
        "near": [[0.0], [0.1]],
        "far": [[10.0], [10.1]],
    }
    target_features = [[0.0], [0.2]]

    full = mmd_source_group_weights(source_features, target_features, gamma="scale", temperature=0.1, blend=1.0)
    blended = mmd_source_group_weights(source_features, target_features, gamma="scale", temperature=0.1, blend=0.25)

    assert full.weights["near"] > blended.weights["near"] > 1.0
    assert full.weights["far"] < blended.weights["far"] < 1.0


def test_mmd_scores_are_negative_mmd_utilities() -> None:
    source_features = {
        "near": [[0.0], [0.1]],
        "far": [[3.0], [3.1]],
    }
    target_features = [[0.05], [0.2]]

    scores = mmd_source_group_scores(source_features, target_features, gamma="scale")

    assert scores["near"] > scores["far"]


def test_mmd_gamma_heuristics_and_validation() -> None:
    source = [np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=float)]
    target = np.asarray([[0.0, 1.0], [1.0, 1.0]], dtype=float)

    assert resolve_mmd_gamma("median", source, target) > 0.0
    assert resolve_mmd_gamma("scale", source, target) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="gamma"):
        resolve_mmd_gamma(0.0, source, target)


def test_mmd_gamma_rejects_unknown_string_with_public_validation_error() -> None:
    source = [np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=float)]
    target = np.asarray([[0.0, 1.0], [1.0, 1.0]], dtype=float)

    with pytest.raises(ValueError, match="gamma must be positive and finite"):
        resolve_mmd_gamma("not-a-gamma", source, target)


def test_mmd_rejects_mismatched_feature_width() -> None:
    with pytest.raises(ValueError, match="feature width"):
        mmd_source_group_weights({"bad": [[0.0, 1.0]]}, [[0.0]])
