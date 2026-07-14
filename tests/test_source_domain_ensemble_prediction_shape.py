from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from neureptrace.decoding.source_ensemble import fit_source_domain_probability_ensemble


class ShortPredictionEstimator(BaseEstimator):
    """Prediction-only estimator that drops the final requested row."""

    def fit(self, features, labels):
        self.classes_ = np.unique(labels)
        return self

    def predict(self, features):
        return np.zeros(max(0, np.asarray(features).shape[0] - 1), dtype=int)


def test_source_domain_ensemble_rejects_short_prediction_output() -> None:
    source_features = np.asarray(
        [
            [-2.0, 0.0],
            [-1.5, 0.1],
            [1.5, -0.1],
            [2.0, 0.0],
        ],
        dtype=float,
    )
    target_features = np.asarray([[-1.7, 0.0], [1.7, 0.0]], dtype=float)

    with pytest.raises(ValueError, match="one label per feature row"):
        fit_source_domain_probability_ensemble(
            source_features=source_features,
            source_labels=[0, 0, 1, 1],
            source_domains=["source", "source", "source", "source"],
            target_features=target_features,
            estimator=ShortPredictionEstimator(),
        )
