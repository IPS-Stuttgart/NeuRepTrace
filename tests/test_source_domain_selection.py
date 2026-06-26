from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_selection import (
    normalize_source_selection_metric,
    select_source_domains_by_target_similarity,
    selected_source_subset,
)


def test_select_source_domain_nearest_to_unlabeled_target() -> None:
    source_features = np.asarray(
        [
            [0.0, 0.0],
            [0.2, -0.1],
            [-0.1, 0.1],
            [8.0, 8.0],
            [8.3, 7.9],
            [7.8, 8.2],
        ]
    )
    source_domains = np.asarray(["near", "near", "near", "far", "far", "far"], dtype=object)
    target_features = np.asarray([[0.1, 0.0], [0.0, 0.2], [-0.2, -0.1]])

    result = select_source_domains_by_target_similarity(source_features, source_domains, target_features, metric="mean", top_k=1)

    assert result.selected_domains == ("near",)
    assert result.metadata["source_selection_protocol_category"] == "2_unlabeled_target_adaptive"
    assert result.metadata["source_selection_uses_target_features"] is True
    assert result.metadata["source_selection_uses_target_labels"] is False
    assert result.metadata["source_selection_valid_for_strict_source_only"] is False
    assert result.selected_mask.tolist() == [True, True, True, False, False, False]
    assert np.all(result.sample_weights[:3] > 0.0)
    assert np.all(result.sample_weights[3:] == 0.0)
    assert np.isclose(np.mean(result.sample_weights[result.selected_mask]), 1.0)
    assert result.domain_distances["near"] < result.domain_distances["far"]
    assert result.domain_scores["near"] > result.domain_scores["far"]


def test_max_distance_keeps_minimum_number_of_domains() -> None:
    source_features = np.asarray([[0.0], [0.1], [10.0], [10.2]])
    source_domains = np.asarray(["a", "a", "b", "b"], dtype=object)
    target_features = np.asarray([[0.05], [0.0]])

    result = select_source_domains_by_target_similarity(
        source_features,
        source_domains,
        target_features,
        metric="mean",
        max_distance=0.001,
        min_selected_domains=1,
    )

    assert result.selected_domains == ("a",)
    assert result.selected_mask.tolist() == [True, True, False, False]


def test_selected_source_subset_returns_filtered_rows_and_weights() -> None:
    source_features = np.asarray([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]])
    source_labels = np.asarray(["x", "y", "x", "y"], dtype=object)
    source_domains = np.asarray(["keep", "keep", "drop", "drop"], dtype=object)
    target_features = np.asarray([[0.0, 0.1], [0.1, 0.1]])

    result = select_source_domains_by_target_similarity(source_features, source_domains, target_features, top_k=1)
    selected_features, selected_labels, weights = selected_source_subset(source_features, source_labels, result)

    assert selected_features.shape == (2, 2)
    assert selected_labels.tolist() == ["x", "y"]
    assert weights.shape == (2,)
    assert np.all(weights > 0.0)


def test_class_balancing_equalizes_selected_class_weight_mass() -> None:
    source_features = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [0.3, 0.0],
            [0.0, 0.2],
            [6.0, 6.0],
            [6.1, 6.0],
        ]
    )
    source_labels = np.asarray(["major", "major", "major", "major", "minor", "major", "minor"], dtype=object)
    source_domains = np.asarray(["near", "near", "near", "near", "near", "far", "far"], dtype=object)
    target_features = np.asarray([[0.0, 0.0], [0.2, 0.1]])

    result = select_source_domains_by_target_similarity(
        source_features,
        source_domains,
        target_features,
        metric="mean",
        top_k=1,
        source_labels=source_labels,
        class_balance=True,
    )

    selected_major_mass = float(np.sum(result.sample_weights[(source_labels == "major") & result.selected_mask]))
    selected_minor_mass = float(np.sum(result.sample_weights[(source_labels == "minor") & result.selected_mask]))
    assert np.isclose(selected_major_mass, selected_minor_mass)
    assert np.isclose(np.mean(result.sample_weights[result.selected_mask]), 1.0)


def test_mmd_metric_and_alias_normalization() -> None:
    assert normalize_source_selection_metric("mean-coral") == "mean_covariance"
    assert normalize_source_selection_metric("rbf_mmd") == "mmd"

    source_features = np.asarray([[0.0], [0.2], [2.0], [2.2]])
    source_domains = np.asarray(["a", "a", "b", "b"], dtype=object)
    target_features = np.asarray([[0.1], [0.0]])
    result = select_source_domains_by_target_similarity(source_features, source_domains, target_features, metric="mmd", top_k=1)

    assert result.selected_domains == ("a",)


def test_target_labels_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        select_source_domains_by_target_similarity(
            [[0.0], [1.0]],
            ["a", "b"],
            [[0.1]],
            target_labels=[0],  # type: ignore[call-arg]
        )
