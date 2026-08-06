from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from neureptrace.decoding.random_subspace import fit_random_subspace_ensemble


class _RecordingProbabilityClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, features, labels, sample_weight=None):
        self.classes_ = np.unique(labels)
        self.sample_weight_ = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        return self

    def predict_proba(self, features):
        probabilities = np.full((np.asarray(features).shape[0], self.classes_.shape[0]), 1.0 / self.classes_.shape[0], dtype=float)
        return probabilities


def _toy_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_features = np.asarray([[-2.0, 0.0], [-1.5, 0.1], [-1.8, -0.1], [1.7, 0.0], [2.0, 0.2], [1.8, -0.1]], dtype=float)
    train_labels = np.asarray(["left", "left", "left", "right", "right", "right"], dtype=object)
    test_features = np.asarray([[-1.7, 0.0], [1.9, 0.0]], dtype=float)
    sample_weight = np.asarray([1.0, 1.5, 2.0, 3.0, 4.0, 5.0], dtype=float)
    return train_features, train_labels, test_features, sample_weight


def test_random_subspace_routes_sample_weights_to_pipeline_final_estimator() -> None:
    train_features, train_labels, test_features, sample_weight = _toy_data()

    result = fit_random_subspace_ensemble(
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        config={"n_estimators": 1, "feature_fraction": 1.0, "random_state": 0},
        estimator=make_pipeline(StandardScaler(), _RecordingProbabilityClassifier()),
        sample_weight=sample_weight,
    )

    fitted_classifier = result.members[0].model.pipeline_.steps[-1][1]
    np.testing.assert_allclose(fitted_classifier.sample_weight_, sample_weight[result.members[0].row_indices])


def test_random_subspace_does_not_force_weights_into_unsupported_pipeline() -> None:
    train_features, train_labels, test_features, sample_weight = _toy_data()

    result = fit_random_subspace_ensemble(
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        config={"n_estimators": 1, "feature_fraction": 1.0, "random_state": 0},
        estimator=make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1)),
        sample_weight=sample_weight,
    )

    assert isinstance(result.members[0].model, Pipeline)
    assert result.predictions.tolist() == ["left", "right"]
