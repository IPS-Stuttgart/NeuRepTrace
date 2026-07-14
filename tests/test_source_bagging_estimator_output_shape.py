from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from neureptrace.decoding.source_bagging import fit_source_bagging_decoder


class _ShortProbabilityEstimator(BaseEstimator):
    classes_ = np.asarray([0, 1])

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "_ShortProbabilityEstimator":
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.asarray([[0.75, 0.25]], dtype=float)


class _ShortDecisionEstimator(BaseEstimator):
    def fit(self, features: np.ndarray, labels: np.ndarray) -> "_ShortDecisionEstimator":
        return self

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        return np.zeros(max(int(features.shape[0]) - 1, 0), dtype=float)


class _ShortPredictionEstimator(BaseEstimator):
    def fit(self, features: np.ndarray, labels: np.ndarray) -> "_ShortPredictionEstimator":
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.zeros(max(int(features.shape[0]) - 1, 0), dtype=int)


@pytest.mark.parametrize(
    "estimator",
    [_ShortProbabilityEstimator(), _ShortDecisionEstimator(), _ShortPredictionEstimator()],
)
def test_source_bagging_rejects_estimator_outputs_with_wrong_row_count(estimator: BaseEstimator) -> None:
    source_features = np.asarray([[-2.0], [-1.0], [1.0], [2.0]], dtype=float)
    source_labels = np.asarray(["left", "left", "right", "right"], dtype=object)
    test_features = np.asarray([[-1.8], [1.8]], dtype=float)

    with pytest.raises(ValueError, match=r"one row per (?:test )?feature row"):
        fit_source_bagging_decoder(
            source_features=source_features,
            source_labels=source_labels,
            test_features=test_features,
            estimator=estimator,
            config={"n_estimators": 1, "random_state": 0},
        )
