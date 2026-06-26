from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_selection import select_source_domains_by_target_similarity


def test_source_selection_accepts_single_column_domain_and_label_vectors() -> None:
    source_features = np.asarray([[0.0], [0.1], [5.0], [5.1]])
    source_domains = np.asarray([["near"], ["near"], ["far"], ["far"]], dtype=object)
    source_labels = np.asarray([["a"], ["b"], ["a"], ["b"]], dtype=object)
    target_features = np.asarray([[0.05], [0.0]])

    result = select_source_domains_by_target_similarity(
        source_features,
        source_domains,
        target_features,
        metric="mean",
        top_k=1,
        source_labels=source_labels,
        class_balance=True,
    )

    assert result.selected_domains == ("near",)
    assert result.selected_mask.tolist() == [True, True, False, False]
    assert np.isclose(np.mean(result.sample_weights[result.selected_mask]), 1.0)


def test_source_selection_rejects_true_domain_matrices() -> None:
    source_features = np.asarray([[0.0], [0.1], [5.0], [5.1]])
    source_domains = np.asarray([["near", "near"], ["far", "far"]], dtype=object)
    target_features = np.asarray([[0.05], [0.0]])

    with pytest.raises(ValueError, match="source_domains must be one-dimensional"):
        select_source_domains_by_target_similarity(
            source_features,
            source_domains,
            target_features,
            metric="mean",
        )


def test_source_selection_rejects_true_label_matrices_when_balancing() -> None:
    source_features = np.asarray([[0.0], [0.1], [5.0], [5.1]])
    source_domains = np.asarray(["near", "near", "far", "far"], dtype=object)
    source_labels = np.asarray([["a", "b"], ["a", "b"]], dtype=object)
    target_features = np.asarray([[0.05], [0.0]])

    with pytest.raises(ValueError, match="source_labels must be one-dimensional"):
        select_source_domains_by_target_similarity(
            source_features,
            source_domains,
            target_features,
            metric="mean",
            source_labels=source_labels,
            class_balance=True,
        )
