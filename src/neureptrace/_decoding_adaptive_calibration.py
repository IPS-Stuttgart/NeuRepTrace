from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

_MARKER = "_neureptrace_adaptive_calibration_installed"


class AdaptiveCalibratedClassifierCV(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator: Any, method: str = "sigmoid", cv: int = 3):
        self.estimator = estimator
        self.method = method
        self.cv = cv

    def fit(self, features, labels, sample_weight=None):
        labels_array = np.asarray(labels)
        classes, counts = np.unique(labels_array, return_counts=True)
        if classes.shape[0] < 2:
            raise ValueError("Calibration requires at least two classes.")
        requested = int(self.cv)
        if requested < 2:
            raise ValueError("Calibration cv must be at least 2.")
        min_count = int(counts.min())
        self.classes_ = classes
        self.requested_calibration_cv_ = requested
        self.min_class_count_ = min_count
        if min_count >= 2:
            self.calibration_cv_ = min(requested, min_count)
            self.used_uncalibrated_fallback_ = False
            self.model_ = self._calibration_factory(clone(self.estimator), method=self.method, cv=self.calibration_cv_)
        else:
            self.calibration_cv_ = 0
            self.used_uncalibrated_fallback_ = True
            self.model_ = clone(self.estimator)
        if sample_weight is None:
            self.model_.fit(features, labels)
        else:
            self.model_.fit(features, labels, sample_weight=sample_weight)
        if hasattr(self.model_, "classes_"):
            self.classes_ = np.asarray(self.model_.classes_)
        if hasattr(self.model_, "n_features_in_"):
            self.n_features_in_ = self.model_.n_features_in_
        return self

    def _model(self):
        if not hasattr(self, "model_"):
            raise RuntimeError("Estimator must be fitted before prediction.")
        return self.model_

    def predict_proba(self, features):
        model = self._model()
        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(features), dtype=float)
            if probabilities.ndim == 2:
                return probabilities
        from neureptrace.decoding import score_to_probabilities

        if hasattr(model, "decision_function"):
            return score_to_probabilities(model.decision_function(features))
        predictions = np.asarray(model.predict(features))
        probabilities = np.zeros((predictions.shape[0], self.classes_.shape[0]), dtype=float)
        lookup = {label: index for index, label in enumerate(self.classes_.tolist())}
        for row_index, label in enumerate(predictions.tolist()):
            probabilities[row_index, lookup[label]] = 1.0
        return probabilities

    def decision_function(self, features):
        model = self._model()
        if hasattr(model, "decision_function"):
            return np.asarray(model.decision_function(features), dtype=float)
        probabilities = np.clip(self.predict_proba(features), 1e-12, 1.0)
        if probabilities.shape[1] == 2:
            return np.log(probabilities[:, 1]) - np.log(probabilities[:, 0])
        return np.log(probabilities)

    def predict(self, features):
        model = self._model()
        if hasattr(model, "predict"):
            return np.asarray(model.predict(features))
        return self.classes_[np.argmax(self.predict_proba(features), axis=1)]


def install() -> None:
    from neureptrace import decoding

    if getattr(decoding, _MARKER, False):
        return
    original_factory = decoding._make_calibrated_classifier
    AdaptiveCalibratedClassifierCV._calibration_factory = staticmethod(original_factory)

    def _make_calibrated_classifier(estimator, *, method: str, cv: int):
        return AdaptiveCalibratedClassifierCV(estimator=estimator, method=method, cv=cv)

    decoding.AdaptiveCalibratedClassifierCV = AdaptiveCalibratedClassifierCV
    decoding._make_calibrated_classifier = _make_calibrated_classifier
    setattr(decoding, _MARKER, True)
