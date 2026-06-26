from __future__ import annotations

import numpy as np

from neureptrace.metrics.ranking import rank_class_scores


def test_rank_class_scores_accepts_numpy_column_vector_labels() -> None:
    scores = np.asarray(
        [
            [0.1, 0.9],
            [0.8, 0.2],
            [0.7, 0.3],
        ],
        dtype=float,
    )
    classes = np.asarray([["left"], ["right"]], dtype=object)
    y_true = np.asarray([["right"], ["left"], ["right"]], dtype=object)

    result = rank_class_scores(scores, classes, y_true, top_k=(1, 2), row_top_k=2)

    assert result["top_k_accuracy"] == {1: 2.0 / 3.0, 2: 1.0}
    assert np.allclose(result["true_label_ranks"], np.asarray([1.0, 1.0, 2.0]))
    assert result["rows"][0]["rank1_class"] == "right"
    assert result["rows"][1]["rank1_class"] == "left"


def test_rank_class_scores_preserves_multi_column_composite_labels() -> None:
    scores = np.asarray([[0.2, 0.8]], dtype=float)
    classes = np.asarray([["subject-a", "stim-left"], ["subject-a", "stim-right"]], dtype=object)
    y_true = np.asarray([["subject-a", "stim-right"]], dtype=object)

    result = rank_class_scores(scores, classes, y_true, top_k=(1,), row_top_k=1)

    assert result["top_k_accuracy"] == {1: 1.0}
    assert result["rows"][0]["rank1_class"] == ("subject-a", "stim-right")


def test_rank_class_scores_preserves_higher_dimensional_composite_labels() -> None:
    scores = np.asarray([[0.1, 0.9]], dtype=float)
    classes = np.asarray(
        [
            [["subject-a", "left"]],
            [["subject-a", "right"]],
        ],
        dtype=object,
    )
    y_true = np.asarray([[["subject-a", "right"]]], dtype=object)

    result = rank_class_scores(scores, classes, y_true, top_k=(1,), row_top_k=1)

    assert result["top_k_accuracy"] == {1: 1.0}
    assert result["rows"][0]["rank1_class"] == ("subject-a", "right")
