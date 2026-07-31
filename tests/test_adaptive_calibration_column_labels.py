from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from neureptrace._decoding_adaptive_calibration import AdaptiveCalibratedClassifierCV


class OneDimensionalTargetClassifier(ClassifierMixin, BaseEstimator):
    """Fallback estimator that enforces sklearn's one-dimensional target contract."""

    def fit(self, features, labels):
        features = np.asarray(features, dtype=float)
        labels = np.asarray(labels)
        if labels.ndim != 1:
            raise ValueError("labels must be one-dimensional")
        self.target_shape_ = labels.shape
        self.classes_ = np.unique(labels)
        self.n_features_in_ = features.shape[1]
        return self

    def predict(self, features):
        features = np.asarray(features, dtype=float)
        return np.repeat(self.classes_[0], features.shape[0])


def test_adaptive_calibration_flattens_scalar_column_labels_for_fallback() -> None:
    features = np.asarray([[-1.0], [0.0], [1.0]])
    labels = np.asarray([[0], [0], [1]])
    model = AdaptiveCalibratedClassifierCV(
        estimator=OneDimensionalTargetClassifier(),
        method="sigmoid",
        cv=3,
    )

    model.fit(features, labels)

    assert model.used_uncalibrated_fallback_ is True
    assert model.model_.target_shape_ == (3,)
    assert model.classes_.tolist() == [0, 1]
