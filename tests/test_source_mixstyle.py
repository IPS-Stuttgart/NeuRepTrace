from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_mixstyle import (
    SOURCE_MIXSTYLE_CATEGORY,
    augment_source_domains_mixstyle,
    mixstyle_row,
    source_mixstyle_config,
)


def _toy_source_data():
    features = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.5],
            [0.5, 1.0],
            [10.0, 10.0],
            [11.0, 10.5],
            [10.5, 11.0],
        ],
        dtype=float,
    )
    labels = np.asarray(["a", "b", "a", "a", "b", "a"], dtype=object)
    domains = np.asarray(["s1", "s1", "s1", "s2", "s2", "s2"], dtype=object)
    return features, labels, domains


def test_source_mixstyle_augments_rows_without_target_information() -> None:
    features, labels, domains = _toy_source_data()

    result = augment_source_domains_mixstyle(
        features,
        labels,
        domains,
        config={"mixes_per_row": 2, "alpha": 0.5, "random_state": 7, "synthetic_weight": 0.25},
    )

    assert result.features.shape == (18, 2)
    assert result.labels.shape == (18,)
    assert result.domain_ids.shape == (18,)
    assert result.n_original == 6
    assert result.n_synthetic == 12
    assert result.synthetic_mask.tolist() == [False] * 6 + [True] * 12
    assert np.allclose(result.features[:6], features)
    assert result.labels[:6].tolist() == labels.tolist()
    assert result.labels[6:].tolist() == np.repeat(labels, 2).tolist()
    assert np.allclose(result.sample_weight[:6], 1.0)
    assert np.allclose(result.sample_weight[6:], 0.25)
    assert result.metadata["source_mixstyle_protocol_category"] == SOURCE_MIXSTYLE_CATEGORY
    assert result.metadata["source_mixstyle_uses_target_features"] is False
    assert result.metadata["source_mixstyle_uses_target_labels"] is False
    assert result.metadata["source_mixstyle_valid_for_strict_source_only"] is True
    assert np.all(np.isfinite(result.features))


def test_source_mixstyle_is_deterministic_for_fixed_seed() -> None:
    features, labels, domains = _toy_source_data()
    config = source_mixstyle_config(mixes_per_row=1, alpha=0.3, random_state=13)

    first = augment_source_domains_mixstyle(features, labels, domains, config=config)
    second = augment_source_domains_mixstyle(features, labels, domains, config=config)

    assert np.allclose(first.features, second.features)
    assert first.labels.tolist() == second.labels.tolist()
    assert np.allclose(first.sample_weight, second.sample_weight)


def test_source_mixstyle_include_original_false_returns_only_synthetic_rows() -> None:
    features, labels, domains = _toy_source_data()

    result = augment_source_domains_mixstyle(
        features,
        labels,
        domains,
        config={"mixes_per_row": 1, "include_original": False, "random_state": 5},
    )

    assert result.features.shape == features.shape
    assert result.n_original == 0
    assert result.n_synthetic == features.shape[0]
    assert np.all(result.synthetic_mask)
    assert result.labels.tolist() == labels.tolist()


def test_source_mixstyle_zero_mixes_returns_original_only() -> None:
    features, labels, domains = _toy_source_data()

    result = augment_source_domains_mixstyle(features, labels, domains, config={"mixes_per_row": 0})

    assert result.features.shape == features.shape
    assert np.allclose(result.features, features)
    assert result.labels.tolist() == labels.tolist()
    assert np.allclose(result.sample_weight, 1.0)
    assert not np.any(result.synthetic_mask)
    assert result.metadata["source_mixstyle_n_synthetic_rows"] == 0


def test_source_mixstyle_requires_multiple_domains_when_generating() -> None:
    features = np.asarray([[0.0], [1.0], [2.0]])
    labels = np.asarray([0, 1, 0])
    domains = np.asarray(["only", "only", "only"], dtype=object)

    with pytest.raises(ValueError, match="at least two source domains"):
        augment_source_domains_mixstyle(features, labels, domains, config={"mixes_per_row": 1})


def test_mixstyle_row_interpolates_domain_statistics() -> None:
    row = np.asarray([1.0, 3.0])
    source_stats = {"domain_id": "a", "mean": [1.0, 1.0], "scale": [1.0, 2.0], "n_rows": 3}
    partner_stats = {"domain_id": "b", "mean": [5.0, 5.0], "scale": [3.0, 4.0], "n_rows": 3}

    mixed = mixstyle_row(row, source_stats=source_stats, partner_stats=partner_stats, lam=0.25, style_strength=1.0)

    # standardized row is [0, 1]; mixed mean=[4,4], mixed scale=[2.5,3.5]
    assert np.allclose(mixed, np.asarray([4.0, 7.5], dtype=np.float32))


def test_style_strength_zero_keeps_rows_unchanged_for_synthetic_copy() -> None:
    features, labels, domains = _toy_source_data()

    result = augment_source_domains_mixstyle(
        features,
        labels,
        domains,
        config={"mixes_per_row": 1, "style_strength": 0.0, "include_original": False, "random_state": 1},
    )

    assert np.allclose(result.features, features)


def test_source_mixstyle_rejects_target_labels_argument() -> None:
    features, labels, domains = _toy_source_data()

    with pytest.raises(TypeError):
        augment_source_domains_mixstyle(
            features,
            labels,
            domains,
            target_labels=labels,  # type: ignore[call-arg]
        )
