from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_selection import select_source_domains_by_target_similarity, selected_source_subset


def _selected_label_mass(labels: list[tuple[str, str]], weights: np.ndarray, mask: np.ndarray, label: tuple[str, str]) -> float:
    return float(sum(weight for value, weight, selected in zip(labels, weights, mask, strict=True) if selected and value == label))


def test_source_selection_preserves_composite_domains_and_labels() -> None:
    source_features = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.1],
            [8.0, 8.0],
            [8.2, 8.1],
        ],
        dtype=float,
    )
    source_domains = [("near", "session-a"), ("near", "session-a"), ("near", "session-a"), ("far", "session-b"), ("far", "session-b")]
    source_labels = [("major", "left"), ("major", "left"), ("minor", "right"), ("major", "left"), ("minor", "right")]
    target_features = np.asarray([[0.0, 0.1], [0.2, 0.0]], dtype=float)

    result = select_source_domains_by_target_similarity(
        source_features,
        source_domains,
        target_features,
        metric="mean",
        top_k=1,
        source_labels=source_labels,
        class_balance=True,
    )

    assert result.selected_domains == (("near", "session-a"),)
    assert result.selected_mask.tolist() == [True, True, True, False, False]
    assert np.isclose(
        _selected_label_mass(source_labels, result.sample_weights, result.selected_mask, ("major", "left")),
        _selected_label_mass(source_labels, result.sample_weights, result.selected_mask, ("minor", "right")),
    )
    assert np.isclose(np.mean(result.sample_weights[result.selected_mask]), 1.0)

    selected_features, selected_labels, weights = selected_source_subset(source_features, source_labels, result)
    assert selected_features.shape == (3, 2)
    assert selected_labels.tolist() == [("major", "left"), ("major", "left"), ("minor", "right")]
    assert np.all(weights > 0.0)
