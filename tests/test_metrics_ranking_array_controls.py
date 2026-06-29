from __future__ import annotations

import numpy as np
import pytest

from neureptrace.metrics.ranking import rank_class_scores


@pytest.mark.parametrize("bad_value", [np.asarray(1), np.array([1]), np.asarray(True), np.array([True])])
def test_rank_class_scores_rejects_array_valued_rank_parameters(bad_value: np.ndarray) -> None:
    scores = [[0.8, 0.2]]
    classes = ["target", "distractor"]
    y_true = ["target"]

    with pytest.raises(ValueError, match="top_k values must be integers"):
        rank_class_scores(scores, classes, y_true, top_k=(bad_value,))

    with pytest.raises(ValueError, match="row_top_k values must be integers"):
        rank_class_scores(scores, classes, y_true, row_top_k=bad_value)
