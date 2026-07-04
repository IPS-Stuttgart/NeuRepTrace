from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_outlier import compute_source_outlier_weights


def test_source_outlier_preserves_tuple_labels_atomically() -> None:
    features = np.asarray([[0.0], [0.2], [2.0], [10.0], [10.2], [13.0]], dtype=float)
    labels = [
        ("face", "seen"),
        ("face", "seen"),
        ("face", "seen"),
        ("object", "new"),
        ("object", "new"),
        ("object", "new"),
    ]

    result = compute_source_outlier_weights(
        features,
        labels,
        config={
            "threshold_mode": "quantile",
            "quantile": 0.75,
            "weight_mode": "binary",
            "use_diagonal_scale": False,
        },
    )

    assert result.classes.shape == (2,)
    assert result.classes.tolist() == [("face", "seen"), ("object", "new")]
    assert set(result.thresholds) == {("face", "seen"), ("object", "new")}
    assert result.sample_weights.shape == (6,)
    assert result.metadata["source_outlier_n_classes"] == 2
    assert "('face', 'seen'):3" in result.metadata["source_outlier_class_counts"]


def test_source_outlier_preserves_row_vector_labels_atomically() -> None:
    features = np.asarray([[0.0], [0.2], [2.0], [10.0], [10.2], [13.0]], dtype=float)
    labels = np.asarray(
        [
            ["face", "seen"],
            ["face", "seen"],
            ["face", "seen"],
            ["object", "new"],
            ["object", "new"],
            ["object", "new"],
        ],
        dtype=object,
    )

    result = compute_source_outlier_weights(
        features,
        labels,
        config={
            "threshold_mode": "quantile",
            "quantile": 0.75,
            "weight_mode": "binary",
            "use_diagonal_scale": False,
        },
    )

    assert result.classes.shape == (2,)
    assert result.classes.tolist() == [("face", "seen"), ("object", "new")]
    assert set(result.thresholds) == {("face", "seen"), ("object", "new")}
    assert result.sample_weights.shape == (6,)
