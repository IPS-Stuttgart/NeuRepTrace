from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_balance import (
    SOURCE_BALANCE_CATEGORY,
    SOURCE_GROUP_CAP_CATEGORY,
    cap_source_rows_per_group,
    compute_source_balance_weights,
    normalize_balance_strategy,
    normalize_balance_target,
    resample_source_rows_balanced,
    source_balance_config,
    source_group_cap_config,
)


def test_class_balance_weights_have_mean_one() -> None:
    labels = np.asarray(["a", "a", "a", "b"], dtype=object)

    result = compute_source_balance_weights(labels, config={"strategy": "class", "target": "max"})

    assert np.isclose(np.mean(result.sample_weights), 1.0)
    assert result.sample_weights[-1] > result.sample_weights[0]
    assert result.group_counts == {"a": 3, "b": 1}
    assert result.metadata["source_balance_protocol_category"] == SOURCE_BALANCE_CATEGORY
    assert result.metadata["source_balance_uses_heldout_features"] is False
    assert result.metadata["source_balance_uses_heldout_labels"] is False
    assert result.metadata["source_balance_valid_for_strict_source_only"] is True


def test_class_domain_balance_groups_labels_and_domains() -> None:
    labels = np.asarray(["a", "a", "a", "b", "b"], dtype=object)
    domains = np.asarray(["s1", "s1", "s2", "s1", "s2"], dtype=object)

    result = compute_source_balance_weights(labels, source_domains=domains)

    assert result.group_counts[("a", "s1")] == 2
    assert result.group_counts[("a", "s2")] == 1
    assert result.sample_weights[2] > result.sample_weights[0]
    assert result.metadata["source_balance_uses_source_domains"] is True


def test_balanced_resampling_oversamples_to_largest_group() -> None:
    features = np.asarray([[0.0], [1.0], [2.0], [10.0]], dtype=float)
    labels = np.asarray(["a", "a", "a", "b"], dtype=object)

    result = resample_source_rows_balanced(features, labels, config={"strategy": "class", "target": "max", "random_state": 4})

    assert result.features.shape == (6, 1)
    assert result.labels.tolist().count("a") == 3
    assert result.labels.tolist().count("b") == 3
    assert result.metadata["source_balance_resampled"] is True
    assert result.metadata["source_balance_n_output_rows"] == 6


def test_balanced_resampling_can_undersample_to_smallest_group() -> None:
    features = np.asarray([[0.0], [1.0], [2.0], [10.0]], dtype=float)
    labels = np.asarray(["a", "a", "a", "b"], dtype=object)

    result = resample_source_rows_balanced(features, labels, config={"strategy": "class", "target": "min", "random_state": 4})

    assert result.features.shape == (2, 1)
    assert result.labels.tolist().count("a") == 1
    assert result.labels.tolist().count("b") == 1


def test_class_group_cap_keeps_at_most_requested_rows_per_class() -> None:
    features = np.arange(8, dtype=float).reshape(8, 1)
    labels = np.asarray(["a", "a", "a", "a", "a", "b", "b", "b"], dtype=object)

    result = cap_source_rows_per_group(features, labels, config={"strategy": "class", "max_rows_per_group": 2, "random_state": 9})

    assert result.features.shape == (4, 1)
    assert result.labels.tolist().count("a") == 2
    assert result.labels.tolist().count("b") == 2
    assert result.group_counts == {"a": 5, "b": 3}
    assert result.group_selected_counts == {"a": 2, "b": 2}
    assert result.metadata["source_group_cap_protocol_category"] == SOURCE_GROUP_CAP_CATEGORY
    assert result.metadata["source_group_cap_uses_heldout_features"] is False
    assert result.metadata["source_group_cap_uses_heldout_labels"] is False
    assert result.metadata["source_group_cap_valid_for_strict_source_only"] is True


def test_class_domain_group_cap_respects_domains_without_shuffle() -> None:
    features = np.arange(7, dtype=float).reshape(7, 1)
    labels = np.asarray(["a", "a", "a", "a", "b", "b", "b"], dtype=object)
    domains = np.asarray(["s1", "s1", "s1", "s2", "s1", "s1", "s2"], dtype=object)

    result = cap_source_rows_per_group(
        features,
        labels,
        source_domains=domains,
        config={"strategy": "class_domain", "max_rows_per_group": 2, "shuffle_within_group": False},
    )

    assert result.source_indices.tolist() == [0, 1, 3, 4, 5, 6]
    assert result.domains is not None
    assert result.domains.tolist() == ["s1", "s1", "s2", "s1", "s1", "s2"]
    assert result.group_selected_counts[("a", "s1")] == 2
    assert result.metadata["source_group_cap_uses_source_domains"] is True


def test_none_strategy_returns_identity_weights_rows_and_cap() -> None:
    features = np.asarray([[0.0], [1.0], [2.0]], dtype=float)
    labels = np.asarray(["a", "a", "b"], dtype=object)

    weights = compute_source_balance_weights(labels, config={"strategy": "none"})
    rows = resample_source_rows_balanced(features, labels, config={"strategy": "none"})
    capped = cap_source_rows_per_group(features, labels, config={"strategy": "none", "max_rows_per_group": 1})

    assert np.allclose(weights.sample_weights, 1.0)
    assert np.allclose(rows.features, features)
    assert rows.labels.tolist() == labels.tolist()
    assert rows.source_indices.tolist() == [0, 1, 2]
    assert np.allclose(capped.features, features)
    assert capped.source_indices.tolist() == [0, 1, 2]


def test_aliases_and_validation() -> None:
    assert normalize_balance_strategy("labels") == "class"
    assert normalize_balance_strategy("domain-class") == "class_domain"
    assert normalize_balance_target("undersample") == "min"
    assert source_balance_config(random_state="7").random_state == 7
    assert source_group_cap_config(max_rows_per_group="3", shuffle_within_group="false").max_rows_per_group == 3
    assert source_group_cap_config(max_rows_per_group="3", shuffle_within_group="false").shuffle_within_group is False

    with pytest.raises(ValueError, match="balance strategy"):
        normalize_balance_strategy("bad")

    with pytest.raises(ValueError, match="balance target"):
        normalize_balance_target("bad")

    with pytest.raises(ValueError, match="max_rows_per_group"):
        source_group_cap_config(max_rows_per_group=0)


def test_heldout_arguments_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        compute_source_balance_weights([0, 1], heldout_labels=[0, 1])  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        cap_source_rows_per_group([[0.0], [1.0]], [0, 1], heldout_features=[[0.5]])  # type: ignore[call-arg]
