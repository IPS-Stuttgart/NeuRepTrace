from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.conditional_subspace import fit_jda, transform_jda


def _composite_label_fixture():
    source_features = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [3.0, 3.0],
            [3.1, 3.0],
        ],
        dtype=float,
    )
    source_labels = [
        np.asarray(["left", 0], dtype=object),
        np.asarray(["left", 0], dtype=object),
        np.asarray(["right", 1], dtype=object),
        np.asarray(["right", 1], dtype=object),
    ]
    target_features = np.asarray(
        [
            [0.05, 0.0],
            [3.05, 3.0],
        ],
        dtype=float,
    )
    return source_features, source_labels, target_features


def test_fit_jda_canonicalizes_composite_array_labels() -> None:
    source_features, source_labels, target_features = _composite_label_fixture()

    result = fit_jda(
        source_features,
        source_labels,
        target_features,
        n_components=1,
        max_iterations=3,
    )

    assert result.source_features.shape == (4, 1)
    assert result.target_features.shape == (2, 1)
    assert result.pseudo_labels.tolist() == [("left", 0), ("right", 1)]
    assert result.metadata["jda_protocol_category"] == "2_unlabeled_target_adaptive"
    assert result.metadata["jda_uses_target_labels"] is False


def test_transform_jda_rejects_feature_width_mismatch() -> None:
    source_features, source_labels, target_features = _composite_label_fixture()
    result = fit_jda(source_features, source_labels, target_features, n_components=1)

    with pytest.raises(ValueError, match="features width"):
        transform_jda(np.ones((2, 3)), result)
