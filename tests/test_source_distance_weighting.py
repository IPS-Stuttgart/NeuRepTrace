from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_distance_weighting import (
    SOURCE_DISTANCE_WEIGHT_CATEGORY,
    compute_source_distance_weights,
    normalize_distance_group_mode,
    source_distance_weight_config,
)


def test_class_distance_weighting_downweights_far_source_rows() -> None:
    features = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [8.0, 8.0],
            [5.0, 5.0],
            [5.1, 5.0],
            [-4.0, -4.0],
        ],
        dtype=float,
    )
    labels = np.asarray(["a", "a", "a", "b", "b", "b"], dtype=object)

    result = compute_source_distance_weights(
        features,
        labels,
        config={"group_mode": "class", "temperature": 0.5, "min_weight": 0.01, "robust": False},
    )

    assert result.sample_weights.shape == (6,)
    assert np.isclose(np.mean(result.sample_weights), 1.0)
    assert result.sample_weights[2] < result.sample_weights[0]
    assert result.sample_weights[5] < result.sample_weights[3]
    assert result.metadata["source_distance_weighting_protocol_category"] == SOURCE_DISTANCE_WEIGHT_CATEGORY
    assert result.metadata["source_distance_weighting_uses_heldout_features"] is False
    assert result.metadata["source_distance_weighting_uses_heldout_labels"] is False
    assert result.metadata["source_distance_weighting_valid_for_strict_source_only"] is True


def test_global_distance_weighting_does_not_need_domains() -> None:
    features = np.asarray([[0.0], [0.1], [0.2], [10.0]], dtype=float)
    labels = np.asarray([0, 0, 1, 1], dtype=object)

    result = compute_source_distance_weights(features, labels, config={"group_mode": "global", "robust": False})

    assert result.metadata["source_distance_weighting_uses_source_labels"] is False
    assert result.metadata["source_distance_weighting_uses_source_domains"] is False
    assert result.sample_weights[-1] < result.sample_weights[0]


def test_class_domain_distance_weighting_uses_domains() -> None:
    features = np.asarray([[0.0], [0.1], [5.0], [5.1]], dtype=float)
    labels = np.asarray(["a", "a", "a", "a"], dtype=object)
    domains = np.asarray(["s1", "s1", "s2", "s2"], dtype=object)

    result = compute_source_distance_weights(features, labels, source_domains=domains, config={"group_mode": "class_domain"})

    assert result.metadata["source_distance_weighting_uses_source_labels"] is True
    assert result.metadata["source_distance_weighting_uses_source_domains"] is True
    assert set(result.group_centers) == {("a", "s1"), ("a", "s2")}


def test_distance_group_aliases_and_validation() -> None:
    assert normalize_distance_group_mode("pooled") == "global"
    assert normalize_distance_group_mode("labels") == "class"
    assert normalize_distance_group_mode("domain-class") == "class_domain"
    cfg = source_distance_weight_config(temperature="2.0", min_weight="0.2", normalize_weights="false")
    assert cfg.temperature == 2.0
    assert cfg.min_weight == 0.2
    assert cfg.normalize_weights is False

    with pytest.raises(ValueError, match="group mode"):
        normalize_distance_group_mode("bad")

    with pytest.raises(ValueError, match="temperature"):
        source_distance_weight_config(temperature=0.0)


def test_distance_weighting_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="source_labels"):
        compute_source_distance_weights([[0.0], [1.0]], [0])

    with pytest.raises(ValueError, match="source_domains"):
        compute_source_distance_weights([[0.0], [1.0]], [0, 1], source_domains=["s1"])


def test_heldout_arguments_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        compute_source_distance_weights(
            [[0.0], [1.0]],
            [0, 1],
            heldout_features=[[0.5]],  # type: ignore[call-arg]
        )
