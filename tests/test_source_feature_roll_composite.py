from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_roll import augment_source_with_feature_roll


def test_source_feature_roll_preserves_composite_labels_and_domains() -> None:
    features = np.asarray([[0.0, 1.0], [1.0, 2.0], [10.0, 11.0], [11.0, 12.0]], dtype=float)
    labels = [("face", "left"), ("face", "left"), ("scene", "right"), ("scene", "right")]
    domains = [("subject-1", "run-1"), ("subject-1", "run-1"), ("subject-2", "run-1"), ("subject-2", "run-1")]

    result = augment_source_with_feature_roll(
        features,
        labels,
        source_domains=domains,
        config={"synthetic_per_class": 1, "max_shift": 1, "random_state": 11},
    )

    assert result.labels.shape == (6,)
    assert result.labels.tolist() == labels + [("face", "left"), ("scene", "right")]
    assert result.metadata["source_feature_roll_n_classes"] == 2
    assert result.metadata["source_feature_roll_n_source_domains"] == 2
    assert result.metadata["source_feature_roll_n_synthetic_rows"] == 2
