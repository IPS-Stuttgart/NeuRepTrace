from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_roll import augment_source_with_feature_roll


def test_source_feature_roll_preserves_matrix_composite_labels_and_domains() -> None:
    features = np.asarray(
        [
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 3.0],
            [10.0, 11.0],
            [11.0, 12.0],
            [12.0, 13.0],
        ],
        dtype=float,
    )
    labels = np.asarray(
        [
            ("face", "early"),
            ("face", "early"),
            ("face", "early"),
            ("house", "late"),
            ("house", "late"),
            ("house", "late"),
        ],
        dtype=object,
    )
    domains = np.asarray(
        [
            ("subject-a", 1),
            ("subject-b", 1),
            ("subject-a", 1),
            ("subject-b", 1),
            ("subject-a", 1),
            ("subject-b", 1),
        ],
        dtype=object,
    )

    assert labels.shape == (6, 2)
    assert domains.shape == (6, 2)

    result = augment_source_with_feature_roll(
        features,
        labels,
        source_domains=domains,
        config={"synthetic_per_class": 1, "max_shift": 1, "random_state": 5},
    )

    assert result.features.shape == (8, 2)
    assert result.labels.shape == (8,)
    assert result.labels.tolist()[:6] == [("face", "early")] * 3 + [("house", "late")] * 3
    assert result.labels.tolist().count(("face", "early")) == 4
    assert result.labels.tolist().count(("house", "late")) == 4
    assert result.metadata["source_feature_roll_n_classes"] == 2
    assert result.metadata["source_feature_roll_n_source_domains"] == 2
    for content_index, label in zip(result.content_indices, result.labels[result.synthetic_mask]):
        assert result.labels[int(content_index)] == label


def test_source_feature_roll_preserves_numeric_matrix_composite_labels() -> None:
    features = np.asarray([[0.0, 1.0], [1.0, 2.0], [10.0, 11.0], [11.0, 12.0]], dtype=float)
    labels = np.asarray([[1, 0], [1, 0], [2, 1], [2, 1]], dtype=int)

    result = augment_source_with_feature_roll(
        features,
        labels,
        config={"synthetic_per_class": 1, "max_shift": 1, "preserve_original": False, "random_state": 2},
    )

    assert result.labels.shape == (2,)
    assert result.labels.tolist().count((1, 0)) == 1
    assert result.labels.tolist().count((2, 1)) == 1
    assert result.metadata["source_feature_roll_n_classes"] == 2
