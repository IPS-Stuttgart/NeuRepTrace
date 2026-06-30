from __future__ import annotations

import numpy as np
import pytest

from neureptrace.metrics.ranking import rank_class_scores


def test_rank_class_scores_accepts_matrix_truth_with_tuple_class_labels() -> None:
    scores = np.asarray(
        [
            [0.2, 0.8],
            [0.7, 0.3],
        ],
        dtype=float,
    )
    classes = [("subject-a", "stim-a"), ("subject-a", "stim-b")]
    y_true = np.asarray(
        [
            ["subject-a", "stim-b"],
            ["subject-a", "stim-a"],
        ],
        dtype=object,
    )

    result = rank_class_scores(scores, classes, y_true, top_k=(1,), row_top_k=1)

    np.testing.assert_array_equal(result["true_label_ranks"], np.asarray([1.0, 1.0]))
    assert result["top_k_accuracy"] == {1: 1.0}
    assert result["rows"][0]["rank1_class"] == ("subject-a", "stim-b")
    assert result["rows"][0]["true_label_score"] == pytest.approx(0.8)


def test_rank_class_scores_matches_matrix_truth_against_list_class_labels() -> None:
    scores = np.asarray(
        [
            [0.2, 0.8],
            [0.7, 0.3],
        ],
        dtype=float,
    )
    classes = [["subject-a", "stim-a"], ["subject-a", "stim-b"]]
    y_true = np.asarray(
        [
            ["subject-a", "stim-b"],
            ["subject-a", "stim-a"],
        ],
        dtype=object,
    )

    result = rank_class_scores(scores, classes, y_true, top_k=(1,), row_top_k=1)

    np.testing.assert_array_equal(result["true_label_ranks"], np.asarray([1.0, 1.0]))
    assert result["top_k_accuracy"] == {1: 1.0}
    assert result["rows"][0]["rank1_class"] == ["subject-a", "stim-b"]
    assert result["rows"][0]["true_label_score"] == pytest.approx(0.8)


def test_rank_class_scores_rejects_matrix_truth_with_mismatched_sequence_label_shape() -> None:
    scores = np.asarray([[0.9, 0.1]], dtype=float)
    classes = [("subject-a", "stim-a", "extra"), ("subject-a", "stim-b", "extra")]
    y_true = np.asarray([["subject-a", "stim-a"]], dtype=object)

    with pytest.raises(ValueError, match="y_true must be one-dimensional"):
        rank_class_scores(scores, classes, y_true, top_k=(1,), row_top_k=1)
