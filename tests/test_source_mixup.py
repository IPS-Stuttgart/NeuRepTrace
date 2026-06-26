from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_mixup import (
    SOURCE_MIXUP_CATEGORY,
    augment_source_with_mixup,
    mixup_rows,
    normalize_hard_label_policy,
    source_mixup_config,
)


def test_mixup_rows_compute_convex_combinations() -> None:
    content = np.asarray([[0.0, 2.0], [10.0, 20.0]])
    partner = np.asarray([[2.0, 4.0], [30.0, 40.0]])

    mixed = mixup_rows(content, partner, lambdas=[0.25, 0.75])

    assert np.allclose(mixed, np.asarray([[1.5, 3.5], [15.0, 25.0]]))


def test_source_mixup_appends_same_class_synthetic_rows() -> None:
    features = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [5.0, 5.0],
            [6.0, 5.0],
            [5.0, 6.0],
        ],
        dtype=float,
    )
    labels = np.asarray(["a", "a", "a", "b", "b", "b"], dtype=object)
    domains = np.asarray(["s1", "s2", "s3", "s1", "s2", "s3"], dtype=object)

    result = augment_source_with_mixup(
        features,
        labels,
        source_domains=domains,
        config={"synthetic_per_class": 2, "same_class_partner": True, "cross_domain_partner": True, "random_state": 7},
    )

    assert result.features.shape == (10, 2)
    assert result.labels.shape == (10,)
    assert result.synthetic_mask.tolist() == [False] * 6 + [True] * 4
    assert result.n_synthetic == 4
    assert result.metadata["source_mixup_protocol_category"] == SOURCE_MIXUP_CATEGORY
    assert result.metadata["source_mixup_uses_target_features"] is False
    assert result.metadata["source_mixup_uses_target_labels"] is False
    assert result.metadata["source_mixup_valid_for_strict_source_only"] is True
    assert np.allclose(result.label_distributions.sum(axis=1), 1.0)
    assert np.all(result.labels[result.synthetic_mask] == labels[result.content_indices])
    assert np.all(domains[result.partner_indices] != domains[result.content_indices])


def test_source_mixup_preserves_tuple_labels_as_atomic_classes() -> None:
    features = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [5.0, 5.0],
            [6.0, 5.0],
        ],
        dtype=float,
    )
    labels = [("cue", "left"), ("cue", "left"), ("cue", "right"), ("cue", "right")]
    domains = np.asarray(["s1", "s2", "s1", "s2"], dtype=object)

    result = augment_source_with_mixup(
        features,
        labels,
        source_domains=domains,
        config={"synthetic_per_class": 1, "same_class_partner": True, "cross_domain_partner": True, "random_state": 11},
    )

    assert result.classes.tolist() == [("cue", "left"), ("cue", "right")]
    assert result.labels[:4].tolist() == labels
    assert result.labels[result.synthetic_mask].tolist() == [labels[index] for index in result.content_indices]
    assert result.label_distributions.shape == (6, 2)
    assert np.allclose(result.label_distributions.sum(axis=1), 1.0)


def test_cross_class_mixup_returns_soft_label_distributions() -> None:
    features = np.asarray([[0.0, 0.0], [10.0, 10.0]], dtype=float)
    labels = np.asarray(["left", "right"], dtype=object)

    result = augment_source_with_mixup(
        features,
        labels,
        config={
            "synthetic_per_class": 1,
            "same_class_partner": False,
            "cross_domain_partner": False,
            "hard_label_policy": "dominant",
            "preserve_original": False,
            "random_state": 13,
        },
    )

    assert result.features.shape == (2, 2)
    assert result.label_distributions.shape == (2, 2)
    assert np.allclose(result.label_distributions.sum(axis=1), 1.0)
    assert np.all(result.label_distributions > 0.0)
    expected_hard = []
    for lam, content_index, partner_index in zip(result.lambdas, result.content_indices, result.partner_indices, strict=True):
        expected_hard.append(labels[content_index] if lam >= 0.5 else labels[partner_index])
    assert result.labels.tolist() == expected_hard


def test_source_mixup_is_reproducible_with_fixed_random_state() -> None:
    features = np.asarray([[0.0], [1.0], [2.0], [3.0], [10.0], [11.0], [12.0], [13.0]], dtype=float)
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    config = source_mixup_config(synthetic_per_class=3, random_state=42)

    first = augment_source_with_mixup(features, labels, config=config)
    second = augment_source_with_mixup(features, labels, config=config)

    assert np.allclose(first.features, second.features)
    assert first.labels.tolist() == second.labels.tolist()
    assert first.content_indices.tolist() == second.content_indices.tolist()
    assert first.partner_indices.tolist() == second.partner_indices.tolist()
    assert np.allclose(first.lambdas, second.lambdas)


def test_disabled_source_mixup_returns_original_rows_only() -> None:
    features = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    labels = np.asarray(["x", "y"], dtype=object)

    result = augment_source_with_mixup(features, labels)

    assert np.allclose(result.features, features)
    assert result.labels.tolist() == labels.tolist()
    assert not np.any(result.synthetic_mask)
    assert result.n_synthetic == 0
    assert np.allclose(result.label_distributions, np.eye(2))


def test_hard_label_policy_aliases_and_validation() -> None:
    assert normalize_hard_label_policy("lambda-dominant") == "dominant"
    assert normalize_hard_label_policy("content_label") == "content"

    with pytest.raises(ValueError, match="hard_label_policy"):
        normalize_hard_label_policy("unknown")


def test_mixup_rows_reject_invalid_lambdas() -> None:
    with pytest.raises(ValueError, match="lambdas"):
        mixup_rows([[0.0]], [[1.0]], lambdas=[1.5])


def test_target_labels_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        augment_source_with_mixup(
            [[0.0], [1.0]],
            [0, 1],
            target_labels=[0, 1],  # type: ignore[call-arg]
        )
