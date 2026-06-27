from __future__ import annotations

import numpy as np
import pytest

from neureptrace.metrics.ranking import rank_class_scores


def test_rank_class_scores_rejects_numpy_y_true_column_vector_labels() -> None:
    scores = np.asarray(
        [
            [0.1, 0.9],
            [0.8, 0.2],
            [0.7, 0.3],
        ],
        dtype=float,
    )
    classes = np.asarray(["class_a", "class_b"], dtype=object)
    y_true = np.asarray([["class_b"], ["class_a"], ["class_b"]], dtype=object)

    with pytest.raises(ValueError, match="y_true must be one-dimensional"):
        rank_class_scores(scores, classes, y_true, top_k=(1, 2), row_top_k=2)


def test_rank_class_scores_rejects_numpy_class_column_vector_labels() -> None:
    scores = np.asarray([[0.1, 0.9]], dtype=float)
    classes = np.asarray([["class_a"], ["class_b"]], dtype=object)
    y_true = np.asarray(["class_b"], dtype=object)

    with pytest.raises(ValueError, match="classes must be one-dimensional"):
        rank_class_scores(scores, classes, y_true, top_k=(1,), row_top_k=2)


def test_rank_class_scores_preserves_multi_column_composite_labels() -> None:
    scores = np.asarray([[0.2, 0.8]], dtype=float)
    classes = np.asarray([["subject-a", "stim-a"], ["subject-a", "stim-b"]], dtype=object)
    y_true = np.asarray([["subject-a", "stim-b"]], dtype=object)

    result = rank_class_scores(scores, classes, y_true, top_k=(1,), row_top_k=1)

    assert result["top_k_accuracy"] == {1: 1.0}
    assert result["rows"][0]["rank1_class"] == ("subject-a", "stim-b")


def test_rank_class_scores_preserves_higher_dimensional_composite_labels() -> None:
    scores = np.asarray([[0.1, 0.9]], dtype=float)
    classes = np.asarray(
        [
            [["subject-a", "stim-a"]],
            [["subject-a", "stim-b"]],
        ],
        dtype=object,
    )
    y_true = np.asarray([[["subject-a", "stim-b"]]], dtype=object)

    result = rank_class_scores(scores, classes, y_true, top_k=(1,), row_top_k=1)

    assert result["top_k_accuracy"] == {1: 1.0}
    assert result["rows"][0]["rank1_class"] == ("subject-a", "stim-b")
