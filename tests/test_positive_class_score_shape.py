from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.classifiers import positive_class_score


class MatrixDecisionModel:
    def decision_function(self, features):
        n_rows = np.asarray(features).shape[0]
        return np.column_stack((np.arange(n_rows, dtype=float), np.arange(n_rows, dtype=float) + 0.5))


class MatrixProbabilityModel:
    def predict_proba(self, features):
        n_rows = np.asarray(features).shape[0]
        probabilities = np.zeros((n_rows, 2), dtype=float)
        probabilities[:, 0] = 0.25
        probabilities[:, 1] = 0.75
        return probabilities


class OneColumnDecisionModel:
    def decision_function(self, features):
        return np.ones((np.asarray(features).shape[0], 1), dtype=float)


def test_positive_class_score_extracts_positive_decision_column():
    features = np.zeros((3, 2), dtype=float)

    scores = positive_class_score(MatrixDecisionModel(), features)

    assert scores.shape == (3,)
    np.testing.assert_allclose(scores, np.asarray([0.5, 1.5, 2.5]))


def test_positive_class_score_extracts_positive_probability_column():
    features = np.zeros((4, 2), dtype=float)

    scores = positive_class_score(MatrixProbabilityModel(), features)

    assert scores.shape == (4,)
    np.testing.assert_allclose(scores, np.full(4, 0.75))


def test_positive_class_score_rejects_single_column_matrix_scores():
    with pytest.raises(ValueError, match="at least two class columns"):
        positive_class_score(OneColumnDecisionModel(), np.zeros((2, 2), dtype=float))
