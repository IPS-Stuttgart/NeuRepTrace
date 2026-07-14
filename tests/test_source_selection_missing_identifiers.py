from __future__ import annotations

import numpy as np

from neureptrace._object_label_utils import label_equal_mask, values_equal
from neureptrace.decoding.source_selection import select_source_domains_by_target_similarity


def test_source_selection_groups_missing_domain_identifiers() -> None:
    source_features = np.asarray([[0.0], [0.2], [10.0], [10.2]], dtype=float)
    source_domains = np.empty(4, dtype=object)
    source_domains[:] = [float("nan"), np.float64("nan"), "far", "far"]
    target_features = np.asarray([[0.05], [0.15]], dtype=float)

    result = select_source_domains_by_target_similarity(
        source_features,
        source_domains,
        target_features,
        metric="mean",
        top_k=1,
    )

    assert len(result.selected_domains) == 1
    assert values_equal(result.selected_domains[0], np.nan)
    assert result.selected_mask.tolist() == [True, True, False, False]
    assert np.all(result.sample_weights[:2] > 0.0)
    assert np.all(result.sample_weights[2:] == 0.0)
    assert np.isclose(np.mean(result.sample_weights[result.selected_mask]), 1.0)
    assert np.isfinite(result.domain_distances[result.selected_domains[0]])
    assert result.metadata["source_selection_n_source_domains"] == 2


def test_source_selection_balances_equivalent_missing_class_labels() -> None:
    source_features = np.asarray([[0.0], [0.1], [0.2], [0.3], [0.4]], dtype=float)
    source_domains = ["near"] * 5
    source_labels = np.empty(5, dtype=object)
    source_labels[:] = [float("nan"), np.float64("nan"), "other", "other", "other"]
    target_features = np.asarray([[0.15], [0.25]], dtype=float)

    result = select_source_domains_by_target_similarity(
        source_features,
        source_domains,
        target_features,
        metric="mean",
        source_labels=source_labels,
        class_balance=True,
    )

    missing_mask = label_equal_mask(source_labels, np.nan)
    other_mask = label_equal_mask(source_labels, "other")
    assert np.isclose(
        np.sum(result.sample_weights[missing_mask]),
        np.sum(result.sample_weights[other_mask]),
    )
    assert np.isclose(np.mean(result.sample_weights), 1.0)
