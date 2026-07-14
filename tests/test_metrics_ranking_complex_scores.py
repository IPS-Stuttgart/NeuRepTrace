from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.metrics.ranking import rank_class_scores


def _nested_score_generators(rows):
    return ((value for value in row) for row in rows)


@pytest.mark.parametrize(
    "scores",
    [
        [[0.9 + 0.1j, 0.1]],
        np.asarray([[0.9 + 0.1j, 0.1]]),
        np.asarray([[0.9 + 0.1j, 0.1]], dtype=object),
        pd.DataFrame([[0.9 + 0.1j, 0.1]], columns=["target", "distractor"]),
        _nested_score_generators([[0.9 + 0.1j, 0.1]]),
    ],
)
def test_rank_class_scores_rejects_complex_scores(scores: object) -> None:
    with pytest.raises(ValueError, match="real-valued scores"):
        rank_class_scores(scores, ["target", "distractor"], ["target"], top_k=(1,))


def test_rank_class_scores_keeps_real_score_behavior() -> None:
    result = rank_class_scores(
        np.asarray([[0.9, 0.1], [0.2, 0.8]]),
        ["target", "distractor"],
        ["target", "distractor"],
        top_k=(1,),
        row_top_k=1,
    )
    np.testing.assert_array_equal(result["true_label_ranks"], np.asarray([1.0, 1.0]))
    assert result["top_k_accuracy"] == {1: pytest.approx(1.0)}
