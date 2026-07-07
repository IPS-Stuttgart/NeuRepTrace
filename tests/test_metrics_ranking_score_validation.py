from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.metrics.ranking import rank_class_scores


def test_rank_class_scores_materializes_one_pass_score_iterables() -> None:
    score_rows = ((value for value in row) for row in ((0.9, 0.1), (0.2, 0.8)))

    result = rank_class_scores(
        score_rows,
        ["target", "distractor"],
        ["target", "distractor"],
        top_k=(1,),
        row_top_k=1,
    )

    np.testing.assert_array_equal(result["true_label_ranks"], np.array([1.0, 1.0]))
    assert result["top_k_accuracy"] == {1: pytest.approx(1.0)}
    assert result["rows"][0]["rank1_class"] == "target"
    assert result["rows"][1]["rank1_class"] == "distractor"


def test_rank_class_scores_preserves_array_like_score_inputs() -> None:
    scores = pd.DataFrame([[0.9, 0.1], [0.2, 0.8]], columns=["target", "distractor"])

    result = rank_class_scores(
        scores,
        ["target", "distractor"],
        ["target", "distractor"],
        top_k=(1,),
        row_top_k=1,
    )

    np.testing.assert_array_equal(result["true_label_ranks"], np.array([1.0, 1.0]))
    assert result["top_k_accuracy"] == {1: pytest.approx(1.0)}


@pytest.mark.parametrize(
    "scores",
    [
        [[True, False]],
        np.array([[True, False]]),
        np.array([[1.0, False]], dtype=object),
        pd.DataFrame([[True, False]], columns=["target", "distractor"]),
        ((value for value in row) for row in ((True, False),)),
    ],
)
def test_rank_class_scores_rejects_boolean_scores(scores: object) -> None:
    with pytest.raises(ValueError, match="scores must contain numeric score values, not boolean flags"):
        rank_class_scores(scores, ["target", "distractor"], ["target"], top_k=(1,))
