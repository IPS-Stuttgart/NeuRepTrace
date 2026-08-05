from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.class_scores import as_class_score_matrix, class_score_matrix


class BooleanBinaryDecisionScores:
    classes_ = np.asarray(["left", "right"])

    def decision_function(self, features):
        n_rows = np.asarray(features).shape[0]
        return np.resize(np.asarray([True, False], dtype=bool), n_rows)


class NonFiniteBinaryDecisionScores:
    classes_ = np.asarray(["left", "right"])

    def __init__(self, value: float):
        self.value = value

    def decision_function(self, features):
        n_rows = np.asarray(features).shape[0]
        return np.full(n_rows, self.value, dtype=float)


def _boolean_score_generator():
    return (value for value in [True, False])


@pytest.mark.parametrize(
    "raw_scores",
    [
        np.asarray([True, False], dtype=bool),
        np.asarray([[True], [False]], dtype=bool),
        _boolean_score_generator(),
    ],
)
def test_as_class_score_matrix_rejects_boolean_scores_before_float_coercion(raw_scores) -> None:
    with pytest.raises(ValueError, match="raw_scores must contain numeric score values, not boolean flags"):
        as_class_score_matrix(raw_scores, ["left", "right"], n_samples=2)


def test_class_score_matrix_rejects_boolean_estimator_outputs() -> None:
    with pytest.raises(ValueError, match="raw_scores must contain numeric score values, not boolean flags"):
        class_score_matrix(
            BooleanBinaryDecisionScores(),
            np.asarray([[-1.0], [2.0]]),
        )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_as_class_score_matrix_rejects_nonfinite_scores(bad_value: float) -> None:
    with pytest.raises(ValueError, match="raw_scores must contain only finite values"):
        as_class_score_matrix(
            np.asarray([bad_value, 1.0]),
            ["left", "right"],
            n_samples=2,
        )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_class_score_matrix_rejects_nonfinite_estimator_outputs(bad_value: float) -> None:
    with pytest.raises(ValueError, match="raw_scores must contain only finite values"):
        class_score_matrix(
            NonFiniteBinaryDecisionScores(bad_value),
            np.asarray([[-1.0], [2.0]]),
        )
