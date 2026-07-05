from __future__ import annotations

import numpy as np

from neureptrace._object_label_utils import values_equal
from neureptrace.decoding.source_interpolation import augment_source_with_interpolation


def test_source_interpolation_preserves_composite_labels_and_prefers_composite_cross_domains() -> None:
    features = np.arange(24, dtype=float).reshape(4, 6)
    labels = [("cat", 1), ("cat", 1), ("dog", 2), ("dog", 2)]
    domains = [("s1", "run1"), ("s1", "run2"), ("s2", "run1"), ("s2", "run2")]

    result = augment_source_with_interpolation(
        features,
        labels,
        source_domains=domains,
        config={"synthetic_per_class": 1, "pair_mode": "same_class_cross_domain", "random_state": 11},
    )

    assert result.features.shape == (6, 6)
    assert result.labels.tolist()[:4] == labels
    assert result.labels.tolist()[4:] == [("cat", 1), ("dog", 2)]
    assert result.metadata["source_interpolation_n_classes"] == 2
    assert result.metadata["source_interpolation_n_source_domains"] == 4
    for content_index, partner_index in zip(result.content_indices, result.partner_indices, strict=True):
        assert values_equal(labels[content_index], labels[partner_index])
        assert not values_equal(domains[content_index], domains[partner_index])


def test_source_interpolation_accepts_matrix_encoded_composite_labels_and_domains() -> None:
    features = np.arange(24, dtype=float).reshape(4, 6)
    labels = np.asarray([[0, 1], [0, 1], [1, 0], [1, 0]], dtype=int)
    domains = np.asarray([[10, 1], [10, 2], [20, 1], [20, 2]], dtype=int)

    result = augment_source_with_interpolation(
        features,
        labels,
        source_domains=domains,
        config={"synthetic_per_class": 1, "pair_mode": "same_class_cross_domain", "random_state": 13},
    )

    assert result.labels.tolist()[:4] == [(0, 1), (0, 1), (1, 0), (1, 0)]
    assert result.labels.tolist()[4:] == [(0, 1), (1, 0)]
    assert result.metadata["source_interpolation_n_classes"] == 2
    assert result.metadata["source_interpolation_n_source_domains"] == 4
