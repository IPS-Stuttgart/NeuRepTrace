from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_domain_subset import apply_source_domain_subset, source_domain_subset_mask


def test_source_domain_subset_mask_is_deterministic() -> None:
    domains = ["a", "a", "b", "b", "c", "c", "d", "d"]

    first = source_domain_subset_mask(domains, omit_fraction=0.5, min_domains=2, random_state=7)
    second = source_domain_subset_mask(domains, omit_fraction=0.5, min_domains=2, random_state=7)

    assert first.selected_mask.tolist() == second.selected_mask.tolist()
    assert first.selected_domains == second.selected_domains
    assert first.omitted_domains == second.omitted_domains
    assert len(first.selected_domains) == 2
    assert len(first.omitted_domains) == 2
    assert int(np.sum(first.selected_mask)) == 4


def test_source_domain_subset_respects_min_domains() -> None:
    domains = ["a", "a", "b", "b", "c", "c"]

    result = source_domain_subset_mask(domains, omit_fraction=1.0, min_domains=2, random_state=1)

    assert len(result.selected_domains) == 2
    assert len(result.omitted_domains) == 1
    assert int(np.sum(result.selected_mask)) == 4


def test_source_domain_subset_preserves_matrix_composite_domains() -> None:
    domains = np.asarray(
        [
            ["subject1", "run1"],
            ["subject1", "run1"],
            ["subject1", "run2"],
            ["subject1", "run2"],
        ],
        dtype=object,
    )

    result = source_domain_subset_mask(domains, omit_fraction=0.5, min_domains=1, random_state=2)

    assert set(result.selected_domains).union(result.omitted_domains) == {("subject1", "run1"), ("subject1", "run2")}
    assert len(result.selected_domains) == 1
    assert len(result.omitted_domains) == 1
    assert int(np.sum(result.selected_mask)) == 2


def test_source_domain_subset_random_state_accepts_scalar_config_values() -> None:
    domains = ["a", "a", "b", "b", "c", "c"]

    reference = source_domain_subset_mask(domains, omit_fraction=0.5, min_domains=2, random_state=7)
    from_string = source_domain_subset_mask(domains, omit_fraction="0.5", min_domains="2", random_state="7")
    from_scalar_array = source_domain_subset_mask(domains, omit_fraction=0.5, min_domains=2, random_state=np.asarray(7))

    assert from_string.selected_mask.tolist() == reference.selected_mask.tolist()
    assert from_string.selected_domains == reference.selected_domains
    assert from_string.omitted_domains == reference.omitted_domains
    assert from_scalar_array.selected_mask.tolist() == reference.selected_mask.tolist()
    assert from_scalar_array.selected_domains == reference.selected_domains
    assert from_scalar_array.omitted_domains == reference.omitted_domains


@pytest.mark.parametrize("seed", [None, "", "none", "null"])
def test_source_domain_subset_random_state_accepts_none_like_values(seed: object) -> None:
    result = source_domain_subset_mask(["a", "b"], omit_fraction=0.0, random_state=seed)

    assert result.selected_mask.tolist() == [True, True]
    assert result.omitted_domains == ()


@pytest.mark.parametrize("seed", [True, np.bool_(False), -1, 1.5, [7], (7,), np.asarray([7])])
def test_source_domain_subset_random_state_rejects_invalid_values(seed: object) -> None:
    with pytest.raises(ValueError, match="random_state"):
        source_domain_subset_mask(["a", "b"], random_state=seed)


def test_apply_source_domain_subset_filters_features_and_labels() -> None:
    features = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    labels = [("x", 1), ("x", 1), ("y", 2), ("y", 2), ("z", 3), ("z", 3)]
    domains = ["a", "a", "b", "b", "c", "c"]

    selected_features, selected_labels, result = apply_source_domain_subset(
        features,
        labels,
        domains,
        omit_fraction=1 / 3,
        min_domains=2,
        random_state=3,
    )

    assert selected_features.shape == (4, 1)
    assert selected_labels.shape == (4,)
    assert len(result.selected_domains) == 2
    assert all(domain in result.selected_domains for domain in np.asarray(domains, dtype=object)[result.selected_mask])


def test_apply_source_domain_subset_preserves_matrix_composite_domains() -> None:
    features = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    labels = ["a", "a", "b", "b"]
    domains = np.asarray(
        [
            ["subject1", "run1"],
            ["subject1", "run1"],
            ["subject1", "run2"],
            ["subject1", "run2"],
        ],
        dtype=object,
    )

    selected_features, selected_labels, result = apply_source_domain_subset(
        features,
        labels,
        domains,
        omit_fraction=0.5,
        min_domains=1,
        random_state=2,
    )

    assert selected_features.shape == (2, 1)
    assert selected_labels.shape == (2,)
    assert set(result.selected_domains).union(result.omitted_domains) == {("subject1", "run1"), ("subject1", "run2")}


def test_source_domain_subset_guardrails() -> None:
    with pytest.raises(ValueError):
        source_domain_subset_mask([], omit_fraction=0.5)
    with pytest.raises(ValueError):
        source_domain_subset_mask(["a", "b"], omit_fraction=-0.1)
    with pytest.raises(ValueError):
        source_domain_subset_mask(["a", "b"], min_domains=3)
    with pytest.raises(ValueError):
        apply_source_domain_subset([[0.0], [1.0]], [0], ["a", "b"])
