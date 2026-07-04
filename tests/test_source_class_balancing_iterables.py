from __future__ import annotations

from collections import Counter

import numpy as np

from neureptrace.decoding.source_balancing import balance_source_classes


def test_source_class_balancing_accepts_one_pass_feature_and_label_iterables() -> None:
    features = (row for row in ([0.0, 0.5], [1.0, 1.5], [10.0, 10.5]))
    labels = (label for label in ["a", "a", "b"])

    result = balance_source_classes(
        features,
        labels,
        config={"mode": "oversample", "target_count": "max", "random_state": 1, "preserve_order": True},
    )

    assert result.features.shape == (4, 2)
    assert result.features.dtype == np.float32
    assert Counter(result.labels.tolist()) == {"a": 2, "b": 2}
    assert result.class_counts_before == {"a": 2, "b": 1}
    assert result.class_counts_after == {"a": 2, "b": 2}
    assert result.synthetic_mask.tolist() == [False, False, False, True]


def test_source_class_balancing_accepts_nested_one_pass_feature_rows() -> None:
    features = ((value for value in row) for row in ([0.0], [1.0], [2.0], [10.0]))
    labels = (label for label in ["a", "a", "a", "b"])

    result = balance_source_classes(features, labels, config={"mode": "weights"})

    assert result.features.shape == (4, 1)
    assert result.labels.tolist() == ["a", "a", "a", "b"]
    assert np.allclose(result.sample_weight, [2 / 3, 2 / 3, 2 / 3, 2.0])
