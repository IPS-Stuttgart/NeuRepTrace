import numpy as np
import pytest

from neureptrace.metrics import rank_class_scores


@pytest.fixture
def ranking_inputs():
    scores = np.array([[0.8, 0.2], [0.3, 0.7]])
    classes = np.array(["left", "right"], dtype=object)
    y_true = np.array(["left", "right"], dtype=object)
    return scores, classes, y_true


def test_rank_class_scores_rejects_complex_numpy_top_k(ranking_inputs):
    scores, classes, y_true = ranking_inputs

    with pytest.raises(ValueError, match="top_k values must be integers"):
        rank_class_scores(
            scores,
            classes,
            y_true,
            top_k=(np.complex128(1.0 + 2.0j),),
        )


def test_rank_class_scores_rejects_complex_numpy_row_top_k(ranking_inputs):
    scores, classes, y_true = ranking_inputs

    with pytest.raises(ValueError, match="row_top_k values must be integers"):
        rank_class_scores(
            scores,
            classes,
            y_true,
            row_top_k=np.complex128(1.0 + 2.0j),
        )
