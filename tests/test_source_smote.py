from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_smote import (
    SOURCE_SMOTE_CATEGORY,
    augment_source_with_smote,
    interpolate_rows,
    source_smote_config,
)


def test_source_smote_appends_same_class_interpolations() -> None:
    features = np.asarray(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [10.0, 10.0],
            [12.0, 10.0],
            [10.0, 12.0],
        ],
        dtype=float,
    )
    labels = np.asarray(["a", "a", "a", "b", "b", "b"], dtype=object)
    domains = np.asarray(["s1", "s2", "s3", "s1", "s2", "s3"], dtype=object)

    result = augment_source_with_smote(
        features,
        labels,
        source_domains=domains,
        config={"synthetic_per_class": 2, "cross_domain_partner": True, "random_state": 7},
    )

    assert result.features.shape == (10, 2)
    assert result.labels.shape == (10,)
    assert result.synthetic_mask.tolist() == [False] * 6 + [True] * 4
    assert result.n_synthetic == 4
    assert result.metadata["source_smote_protocol_category"] == SOURCE_SMOTE_CATEGORY
    assert result.metadata["source_smote_uses_heldout_features"] is False
    assert result.metadata["source_smote_uses_heldout_labels"] is False
    assert result.metadata["source_smote_valid_for_strict_source_only"] is True
    assert np.all(labels[result.content_indices] == result.labels[result.synthetic_mask])
    assert np.all(labels[result.partner_indices] == result.labels[result.synthetic_mask])
    assert np.all(domains[result.partner_indices] != domains[result.content_indices])


def test_source_smote_preserves_composite_labels_and_domains() -> None:
    features = np.arange(20, dtype=float).reshape(4, 5)
    labels = [("cat", 1), ("cat", 1), ("dog", 2), ("dog", 2)]
    domains = [("s1", "run1"), ("s1", "run2"), ("s2", "run1"), ("s2", "run2")]

    result = augment_source_with_smote(
        features,
        labels,
        source_domains=domains,
        config={"synthetic_per_class": 1, "cross_domain_partner": True, "random_state": 11},
    )

    assert result.features.shape == (6, 5)
    assert result.labels.shape == (6,)
    assert result.labels.tolist()[:4] == labels
    assert result.labels.tolist()[4:] == [("cat", 1), ("dog", 2)]
    assert result.metadata["source_smote_n_classes"] == 2
    assert result.metadata["source_smote_n_source_domains"] == 4
    for content_index, partner_index in zip(result.content_indices, result.partner_indices, strict=True):
        assert labels[content_index] == labels[partner_index]
        assert domains[content_index] != domains[partner_index]


def test_interpolate_rows_returns_convex_interpolation() -> None:
    row = interpolate_rows([0.0, 2.0], [4.0, 6.0], 0.25)

    assert np.allclose(row, np.asarray([1.0, 3.0]))


def test_preserve_original_false_returns_only_synthetic_rows() -> None:
    features = np.asarray([[0.0], [1.0], [2.0]], dtype=float)
    labels = np.asarray(["x", "x", "x"], dtype=object)

    result = augment_source_with_smote(
        features,
        labels,
        config={"synthetic_per_class": 3, "preserve_original": False, "random_state": 3},
    )

    assert result.features.shape == (3, 1)
    assert result.labels.tolist() == ["x", "x", "x"]
    assert result.synthetic_mask.tolist() == [True, True, True]


def test_disabled_smote_returns_original_rows_only() -> None:
    features = np.asarray([[0.0], [1.0]], dtype=float)
    labels = np.asarray([0, 1])

    result = augment_source_with_smote(features, labels)

    assert np.allclose(result.features, features)
    assert result.labels.tolist() == [0, 1]
    assert not np.any(result.synthetic_mask)
    assert result.n_synthetic == 0


def test_smote_reproducible_with_fixed_seed() -> None:
    features = np.arange(12, dtype=float).reshape(6, 2)
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    config = source_smote_config(synthetic_per_class=3, random_state=42)

    first = augment_source_with_smote(features, labels, config=config)
    second = augment_source_with_smote(features, labels, config=config)

    assert np.allclose(first.features, second.features)
    assert first.labels.tolist() == second.labels.tolist()
    assert first.content_indices.tolist() == second.content_indices.tolist()
    assert first.partner_indices.tolist() == second.partner_indices.tolist()
    assert np.allclose(first.lambdas, second.lambdas)


def test_invalid_interpolation_weight_is_rejected() -> None:
    with pytest.raises(ValueError, match="lam"):
        interpolate_rows([0.0], [1.0], 1.5)
