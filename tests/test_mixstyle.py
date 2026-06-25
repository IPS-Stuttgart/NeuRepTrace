from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.mixstyle import (
    SOURCE_MIXSTYLE_CATEGORY,
    augment_source_mixstyle,
    augment_source_mixstyle_from_config,
    normalize_mixstyle_domain_pairing,
    source_mixstyle_config,
)


def _toy_sources():
    features = np.asarray(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [0.4, -0.1],
            [5.0, 5.0],
            [5.2, 5.1],
            [4.8, 5.2],
        ],
        dtype=float,
    )
    labels = np.asarray(["a", "b", "a", "a", "b", "a"], dtype=object)
    domains = np.asarray(["s1", "s1", "s1", "s2", "s2", "s2"], dtype=object)
    return features, labels, domains


def test_mixstyle_adds_source_only_synthetic_rows() -> None:
    features, labels, domains = _toy_sources()

    result = augment_source_mixstyle(
        features,
        labels,
        domains,
        augmentations_per_row=2,
        alpha=0.4,
        random_state=7,
    )

    assert result.features.shape == (18, 2)
    assert result.labels.shape == (18,)
    assert result.domains.shape == (18,)
    assert result.synthetic_mask.shape == (18,)
    assert result.n_original == 6
    assert result.n_synthetic == 12
    assert result.labels[:6].tolist() == labels.tolist()
    assert result.labels[6:].tolist() == np.repeat(labels, 2).tolist()
    assert np.all(result.synthetic_mask[:6] == 0)
    assert np.all(result.synthetic_mask[6:] == 1)
    assert result.metadata["source_mixstyle_protocol_category"] == SOURCE_MIXSTYLE_CATEGORY
    assert result.metadata["source_mixstyle_uses_target_features"] is False
    assert result.metadata["source_mixstyle_uses_target_labels"] is False
    assert result.metadata["source_mixstyle_valid_for_strict_source_only"] is True


def test_mixstyle_is_deterministic_for_seed() -> None:
    features, labels, domains = _toy_sources()

    first = augment_source_mixstyle(features, labels, domains, augmentations_per_row=1, random_state=13)
    second = augment_source_mixstyle(features, labels, domains, augmentations_per_row=1, random_state=13)

    assert np.allclose(first.features, second.features)
    assert first.labels.tolist() == second.labels.tolist()
    assert first.domains.tolist() == second.domains.tolist()


def test_mixstyle_supports_synthetic_only_output() -> None:
    features, labels, domains = _toy_sources()

    result = augment_source_mixstyle(features, labels, domains, augmentations_per_row=1, include_original=False, random_state=3)

    assert result.features.shape == features.shape
    assert np.all(result.synthetic_mask)
    assert result.labels.tolist() == labels.tolist()
    assert result.metadata["source_mixstyle_include_original"] is False


def test_mixstyle_rejects_no_output_rows() -> None:
    features, labels, domains = _toy_sources()

    with pytest.raises(ValueError, match="No rows"):
        augment_source_mixstyle(features, labels, domains, augmentations_per_row=0, include_original=False)


def test_mixstyle_rejects_single_domain_when_augmenting() -> None:
    features = np.asarray([[0.0], [1.0], [2.0]])
    labels = np.asarray([0, 1, 0])
    domains = np.asarray(["only", "only", "only"], dtype=object)

    with pytest.raises(ValueError, match="at least two source domains"):
        augment_source_mixstyle(features, labels, domains, augmentations_per_row=1)


def test_mixstyle_preserve_domain_mean_keeps_rows_near_source_domain() -> None:
    features, labels, domains = _toy_sources()

    result = augment_source_mixstyle(
        features,
        labels,
        domains,
        augmentations_per_row=1,
        preserve_domain_mean=True,
        domain_pairing="farthest",
        random_state=4,
    )

    synthetic = result.features[result.synthetic_mask]
    original_domain_means = {domain: features[domains == domain].mean(axis=0) for domain in np.unique(domains)}
    for row_index, domain in enumerate(domains):
        source_center = original_domain_means[domain]
        partner_center = original_domain_means["s2" if domain == "s1" else "s1"]
        assert np.linalg.norm(synthetic[row_index] - source_center) < np.linalg.norm(synthetic[row_index] - partner_center)


def test_mixstyle_class_conditional_path_runs_with_sparse_label_domain_cells() -> None:
    features, labels, domains = _toy_sources()

    result = augment_source_mixstyle(
        features,
        labels,
        domains,
        augmentations_per_row=1,
        class_conditional=True,
        domain_pairing="nearest",
        random_state=9,
    )

    assert result.features.shape == (12, 2)
    assert result.metadata["source_mixstyle_class_conditional"] is True
    assert np.all(np.isfinite(result.features))


def test_mixstyle_preserves_composite_labels_and_domains() -> None:
    features, _, _ = _toy_sources()
    labels = [("a", "left"), ("b", "right"), ("a", "left"), ("a", "left"), ("b", "right"), ("a", "left")]
    domains = [("s1", "run-a"), ("s1", "run-a"), ("s1", "run-a"), ("s2", "run-b"), ("s2", "run-b"), ("s2", "run-b")]

    result = augment_source_mixstyle(
        features,
        labels,
        domains,
        augmentations_per_row=1,
        class_conditional=True,
        domain_pairing="nearest",
        random_state=11,
    )

    assert result.features.shape == (12, 2)
    assert result.labels.shape == (12,)
    assert result.domains.shape == (12,)
    assert result.labels[:6].tolist() == labels
    assert result.labels[6:].tolist() == labels
    assert result.domains[:6].tolist() == domains
    assert all(str(domain).startswith("mixstyle:") for domain in result.domains[6:].tolist())
    assert result.metadata["source_mixstyle_n_classes"] == 2
    assert result.metadata["source_mixstyle_n_domains"] == 2


def test_mixstyle_from_config_mapping() -> None:
    features, labels, domains = _toy_sources()
    cfg = {"augmentations_per_row": 1, "alpha": 0.5, "random_state": 2, "domain_pairing": "random"}

    result = augment_source_mixstyle_from_config(features, labels, domains, cfg)

    assert result.features.shape == (12, 2)
    assert result.metadata["source_mixstyle_domain_pairing"] == "shuffle"


def test_mixstyle_config_normalizes_aliases() -> None:
    config = source_mixstyle_config(domain_pairing="most-different", augmentations_per_row="2", alpha="0.3")

    assert config.domain_pairing == "farthest"
    assert config.augmentations_per_row == 2
    assert np.isclose(config.alpha, 0.3)
    assert normalize_mixstyle_domain_pairing("nearest-domain") == "nearest"


def test_mixstyle_public_api_rejects_target_labels_argument() -> None:
    features, labels, domains = _toy_sources()

    with pytest.raises(TypeError):
        augment_source_mixstyle(
            features,
            labels,
            domains,
            target_labels=[0, 1],  # type: ignore[call-arg]
        )
