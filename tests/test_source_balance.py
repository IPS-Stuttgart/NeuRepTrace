from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_balance import (
    SOURCE_BALANCE_CATEGORY,
    SOURCE_LABEL_SMOOTHING_CATEGORY,
    compute_source_balance_weights,
    normalize_balance_strategy,
    normalize_balance_target,
    normalize_label_smoothing_prior,
    resample_source_rows_balanced,
    smooth_source_labels,
    source_balance_config,
    source_label_prior,
    source_label_smoothing_config,
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


def test_none_strategy_returns_identity_weights_and_rows() -> None:
    features = np.asarray([[0.0], [1.0], [2.0]], dtype=float)
    labels = np.asarray(["a", "a", "b"], dtype=object)

    weights = compute_source_balance_weights(labels, config={"strategy": "none"})
    rows = resample_source_rows_balanced(features, labels, config={"strategy": "none"})

    assert np.allclose(weights.sample_weights, 1.0)
    assert np.allclose(rows.features, features)
    assert rows.labels.tolist() == labels.tolist()
    assert rows.source_indices.tolist() == [0, 1, 2]


def test_source_label_smoothing_uniform_prior() -> None:
    labels = np.asarray(["a", "b", "a"], dtype=object)

    result = smooth_source_labels(labels, config={"smoothing": 0.2, "prior": "uniform"})

    assert result.classes.tolist() == ["a", "b"]
    assert result.distributions.shape == (3, 2)
    assert np.allclose(result.distributions.sum(axis=1), 1.0)
    assert np.allclose(result.distributions[0], np.asarray([0.9, 0.1]))
    assert np.allclose(result.prior_distribution, np.asarray([0.5, 0.5]))
    assert result.metadata["source_label_smoothing_protocol_category"] == SOURCE_LABEL_SMOOTHING_CATEGORY
    assert result.metadata["source_label_smoothing_uses_heldout_features"] is False
    assert result.metadata["source_label_smoothing_uses_heldout_labels"] is False
    assert result.metadata["source_label_smoothing_valid_for_strict_source_only"] is True


def test_source_label_smoothing_empirical_prior_and_explicit_class_order() -> None:
    labels = np.asarray(["a", "a", "b"], dtype=object)

    result = smooth_source_labels(labels, classes=["b", "a"], config={"smoothing": 0.3, "prior": "empirical"})

    assert result.classes.tolist() == ["b", "a"]
    assert np.allclose(result.prior_distribution, np.asarray([1.0 / 3.0, 2.0 / 3.0]))
    assert np.allclose(result.distributions[0], np.asarray([0.1, 0.9]))
    assert np.allclose(result.distributions[2], np.asarray([0.8, 0.2]))


def test_source_label_prior_aliases_and_validation() -> None:
    assert normalize_label_smoothing_prior("balanced") == "uniform"
    assert normalize_label_smoothing_prior("frequency") == "empirical"
    assert source_label_smoothing_config(smoothing="0.25").smoothing == 0.25
    assert np.allclose(source_label_prior([0, 0, 1], prior="empirical"), np.asarray([2.0 / 3.0, 1.0 / 3.0]))

    with pytest.raises(ValueError, match="label smoothing prior"):
        normalize_label_smoothing_prior("bad")

    with pytest.raises(ValueError, match="smoothing"):
        source_label_smoothing_config(smoothing=1.5)

    with pytest.raises(ValueError, match="absent from classes"):
        smooth_source_labels(["a", "b"], classes=["a"])


def test_aliases_and_validation() -> None:
    assert normalize_balance_strategy("labels") == "class"
    assert normalize_balance_strategy("domain-class") == "class_domain"
    assert normalize_balance_target("undersample") == "min"
    assert source_balance_config(random_state="7").random_state == 7

    with pytest.raises(ValueError, match="balance strategy"):
        normalize_balance_strategy("bad")

    with pytest.raises(ValueError, match="balance target"):
        normalize_balance_target("bad")


def test_heldout_arguments_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        compute_source_balance_weights([0, 1], heldout_labels=[0, 1])  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        smooth_source_labels([0, 1], heldout_labels=[0, 1])  # type: ignore[call-arg]
