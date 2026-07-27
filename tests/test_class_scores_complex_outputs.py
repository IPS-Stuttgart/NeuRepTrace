from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.class_scores import as_class_score_matrix, class_score_matrix


class ComplexBinaryDecisionScores:
    classes_ = np.asarray(["left", "right"])

    def decision_function(self, features):
        n_rows = np.asarray(features).shape[0]
        return np.full(n_rows, 1.0 + 2.0j, dtype=np.complex128)


def _complex_score_generator():
    return (value for value in [1.0 + 2.0j, -1.0 + 3.0j])


@pytest.mark.parametrize(
    "raw_scores",
    [
        np.asarray([1.0 + 2.0j, -1.0 + 3.0j], dtype=np.complex128),
        np.asarray([[1.0 + 2.0j], [-1.0 + 3.0j]], dtype=np.complex128),
        _complex_score_generator(),
    ],
)
def test_as_class_score_matrix_rejects_complex_scores_before_float_coercion(raw_scores) -> None:
    with pytest.raises(ValueError, match="raw_scores must contain real-valued scores"):
        as_class_score_matrix(raw_scores, ["left", "right"], n_samples=2)


def test_class_score_matrix_rejects_complex_estimator_outputs() -> None:
    with pytest.raises(ValueError, match="raw_scores must contain real-valued scores"):
        class_score_matrix(
            ComplexBinaryDecisionScores(),
            np.asarray([[-1.0], [2.0]]),
        )
