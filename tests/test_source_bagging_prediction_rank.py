from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from neureptrace.decoding.source_bagging import fit_source_bagging_decoder


class _ColumnPredictionEstimator(BaseEstimator):
    def fit(self, features: np.ndarray, labels: np.ndarray) -> "_ColumnPredictionEstimator":
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.zeros((features.shape[0], 1), dtype=int)


def test_source_bagging_rejects_non_vector_predictions() -> None:
    source_features = np.asarray([[-2.0], [-1.0], [1.0], [2.0]], dtype=float)
    source_labels = np.asarray(["left", "left", "right", "right"], dtype=object)
    test_features = np.asarray([[-1.8], [1.8]], dtype=float)

    with pytest.raises(ValueError, match="predictions must be one-dimensional"):
        fit_source_bagging_decoder(
            source_features=source_features,
            source_labels=source_labels,
            test_features=test_features,
            estimator=_ColumnPredictionEstimator(),
            config={"n_estimators": 1, "random_state": 0},
        )
