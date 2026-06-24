import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from neureptrace._decoding_adaptive_calibration import AdaptiveCalibratedClassifierCV


class SampleWeightRequiredClassifier(ClassifierMixin, BaseEstimator):
    """Tiny test estimator that proves sample weights reached the final step."""

    def fit(self, features, labels, sample_weight=None):
        if sample_weight is None:
            raise ValueError("sample_weight required")
        features = np.asarray(features, dtype=float)
        self.classes_ = np.unique(labels)
        self.sample_weight_ = np.asarray(sample_weight, dtype=float)
        self.n_features_in_ = features.shape[1]
        return self

    def predict(self, features):
        features = np.asarray(features, dtype=float)
        return np.repeat(self.classes_[0], features.shape[0])

    def predict_proba(self, features):
        predictions = self.predict(features)
        probabilities = np.zeros((predictions.shape[0], self.classes_.shape[0]), dtype=float)
        probabilities[:, 0] = 1.0
        return probabilities


def _tiny_fallback_data():
    return (
        np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]], dtype=float),
        np.array([0, 0, 1]),
        np.array([1.0, 2.0, 3.0], dtype=float),
    )


def test_adaptive_calibration_routes_sample_weight_through_pipeline_fallback() -> None:
    features, labels, sample_weight = _tiny_fallback_data()
    estimator = make_pipeline(StandardScaler(), SampleWeightRequiredClassifier())
    model = AdaptiveCalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=3)

    model.fit(features, labels, sample_weight=sample_weight)

    assert model.used_uncalibrated_fallback_ is True
    assert model.calibration_cv_ == 0
    final_estimator = model.model_.named_steps["sampleweightrequiredclassifier"]
    assert np.array_equal(final_estimator.sample_weight_, sample_weight)
    assert model.predict_proba(features[:2]).shape == (2, 2)


def test_adaptive_calibration_rejects_malformed_sample_weight() -> None:
    features, labels, _sample_weight = _tiny_fallback_data()
    estimator = make_pipeline(StandardScaler(), SampleWeightRequiredClassifier())
    model = AdaptiveCalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=3)

    with pytest.raises(ValueError, match="one weight per label"):
        model.fit(features, labels, sample_weight=[1.0, 2.0])


@pytest.mark.parametrize("bad_cv", [True, False, 1, 2.5, np.nan, np.inf, "3.5", object()])
def test_adaptive_calibration_rejects_invalid_cv_values(bad_cv) -> None:
    features, labels, _sample_weight = _tiny_fallback_data()
    estimator = make_pipeline(StandardScaler(), SampleWeightRequiredClassifier())
    model = AdaptiveCalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=bad_cv)

    with pytest.raises(ValueError, match="Calibration cv must be an integer at least 2"):
        model.fit(features, labels)
