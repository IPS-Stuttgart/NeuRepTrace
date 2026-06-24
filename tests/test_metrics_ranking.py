from __future__ import annotations

import numpy as np
import pytest

from neureptrace.metrics.ranking import rank_class_scores


def test_rank_class_scores_reports_true_label_ranks_and_top_hits() -> None:
    result = rank_class_scores(
        np.array(
            [
                [0.1, 0.7, 0.2],
                [0.6, 0.3, 0.1],
                [0.1, 0.2, 0.7],
            ]
        ),
        ["cat", "dog", "fox"],
        ["dog", "fox", "cat"],
        top_k=(1, 2),
        row_top_k=2,
    )

    np.testing.assert_array_equal(result["true_label_ranks"], np.array([1.0, 3.0, 3.0]))
    assert result["top_k_accuracy"] == {1: pytest.approx(1 / 3), 2: pytest.approx(1 / 3)}
    assert result["mean_true_label_rank"] == pytest.approx(7 / 3)
    assert result["median_true_label_rank"] == pytest.approx(3.0)
    assert result["rows"][0] == {
        "true_label_rank": 1.0,
        "true_label_score": 0.7,
        "rank1_class": "dog",
        "rank1_score": 0.7,
        "rank2_class": "fox",
        "rank2_score": 0.2,
    }


def test_rank_class_scores_matches_nan_class_labels() -> None:
    result = rank_class_scores(
        np.array(
            [
                [0.9, 0.1],
                [0.2, 0.8],
            ]
        ),
        [np.nan, "known"],
        [np.nan, "known"],
        top_k=(1,),
    )

    np.testing.assert_array_equal(result["true_label_ranks"], np.array([1.0, 1.0]))
    assert result["top_k_accuracy"] == {1: pytest.approx(1.0)}
    assert result["rows"][0]["true_label_score"] == pytest.approx(0.9)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_rank_class_scores_rejects_nonfinite_scores(bad_value: float) -> None:
    scores = np.array([[0.5, bad_value]])

    with pytest.raises(ValueError, match="scores must contain only finite values"):
        rank_class_scores(scores, ["target", "distractor"], ["target"])


def test_rank_class_scores_rejects_duplicate_classes() -> None:
    with pytest.raises(ValueError, match="classes must be unique"):
        rank_class_scores([[0.8, 0.2]], ["target", "target"], ["target"])


def test_rank_class_scores_rejects_multidimensional_y_true() -> None:
    with pytest.raises(ValueError, match="y_true must be one-dimensional"):
        rank_class_scores(
            np.array([[0.9, 0.1], [0.2, 0.8]]),
            ["target", "distractor"],
            np.array([["target"], ["distractor"]]),
        )


def test_rank_class_scores_rejects_multidimensional_classes() -> None:
    with pytest.raises(ValueError, match="classes must be one-dimensional"):
        rank_class_scores(
            np.array([[0.9, 0.1]]),
            np.array([["target", "distractor"]]),
            ["target"],
        )


def test_rank_class_scores_allows_empty_class_axis() -> None:
    result = rank_class_scores(np.empty((2, 0)), [], ["a", "b"], top_k=(1,))

    assert np.isnan(result["top_k_accuracy"][1])
    assert np.isnan(result["mean_true_label_rank"])
    assert result["rows"] == [{}, {}]


def test_rank_class_scores_rejects_fractional_rank_parameters() -> None:
    scores = [[0.8, 0.2]]
    classes = ["target", "distractor"]
    y_true = ["target"]

    with pytest.raises(ValueError, match="top_k values must be integers"):
        rank_class_scores(scores, classes, y_true, top_k=(1.5,))

    with pytest.raises(ValueError, match="row_top_k values must be integers"):
        rank_class_scores(scores, classes, y_true, row_top_k=1.5)
