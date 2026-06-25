from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_domain_mask import apply_source_domain_mask, source_domain_mask


def test_source_domain_mask_keeps_string_domain_labels() -> None:
    domains = ["subject_one", "subject_one", "subject_two", "subject_two", "subject_three", "subject_three"]

    result = source_domain_mask(domains, holdout_fraction=1 / 3, min_selected_domains=2, random_state=3)

    assert result.selected_mask.shape == (6,)
    assert len(result.selected_domains) == 2
    assert len(result.heldout_domains) == 1
    assert set(result.selected_domains) | set(result.heldout_domains) == {"subject_one", "subject_two", "subject_three"}


def test_apply_source_domain_mask_filters_features_and_labels() -> None:
    features = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    labels = [("label", 0), ("label", 0), ("label", 1), ("label", 1), ("label", 2), ("label", 2)]
    domains = ["a", "a", "b", "b", "c", "c"]

    selected_features, selected_labels, result = apply_source_domain_mask(
        features,
        labels,
        domains,
        holdout_fraction=1 / 3,
        min_selected_domains=2,
        random_state=3,
    )

    assert selected_features.shape == (4, 1)
    assert selected_labels.shape == (4,)
    assert all(domain in result.selected_domains for domain in np.asarray(domains, dtype=object)[result.selected_mask])


def test_source_domain_mask_preserves_rectangular_composite_domains() -> None:
    domains = np.asarray(
        [
            ["subject_one", "session_a"],
            ["subject_one", "session_a"],
            ["subject_two", "session_a"],
            ["subject_two", "session_a"],
            ["subject_three", "session_b"],
            ["subject_three", "session_b"],
        ],
        dtype=object,
    )

    result = source_domain_mask(domains, holdout_fraction=1 / 3, min_selected_domains=2, random_state=3)

    all_domains = set(result.selected_domains) | set(result.heldout_domains)
    assert result.selected_mask.shape == (6,)
    assert all(isinstance(domain, tuple) for domain in all_domains)
    assert all_domains == {
        ("subject_one", "session_a"),
        ("subject_two", "session_a"),
        ("subject_three", "session_b"),
    }


def test_apply_source_domain_mask_preserves_rectangular_composite_labels() -> None:
    features = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    labels = np.asarray(
        [
            ["left", 0],
            ["left", 0],
            ["right", 1],
            ["right", 1],
            ["rest", 2],
            ["rest", 2],
        ],
        dtype=object,
    )
    domains = np.asarray(
        [
            ["subject_one", "session_a"],
            ["subject_one", "session_a"],
            ["subject_two", "session_a"],
            ["subject_two", "session_a"],
            ["subject_three", "session_b"],
            ["subject_three", "session_b"],
        ],
        dtype=object,
    )

    selected_features, selected_labels, result = apply_source_domain_mask(
        features,
        labels,
        domains,
        holdout_fraction=1 / 3,
        min_selected_domains=2,
        random_state=3,
    )

    assert selected_features.shape == (4, 1)
    assert selected_labels.shape == (4,)
    assert all(isinstance(label, tuple) for label in selected_labels.tolist())
    assert all(isinstance(domain, tuple) for domain in result.selected_domains + result.heldout_domains)


def test_source_domain_mask_rejects_scalar_string_vectors() -> None:
    with pytest.raises(ValueError, match="source_domains"):
        source_domain_mask("subject_one")
    with pytest.raises(ValueError, match="labels"):
        apply_source_domain_mask([[0.0]], "label", ["subject_one"])
    with pytest.raises(ValueError, match="source_domains"):
        apply_source_domain_mask([[0.0]], ["label"], "subject_one")
