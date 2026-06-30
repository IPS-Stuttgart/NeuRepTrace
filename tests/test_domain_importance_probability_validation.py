from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from neureptrace.decoding.domain_importance import fit_domain_classifier_importance_weights


class _StaticDomainEstimator(BaseEstimator):
    def __init__(self, probabilities, classes=(0, 1)):
        self.probabilities = probabilities
        self.classes = classes

    def fit(self, X, y):
        self.classes_ = np.asarray(self.classes)
        return self


def _static_probabilities(self, X):
    rows = np.asarray(self.probabilities, dtype=float)
    if rows.ndim != 2:
        return rows
    if rows.shape[0] == X.shape[0]:
        return rows
    if rows.shape[0] == 1:
        return np.repeat(rows, X.shape[0], axis=0)
    return rows[: X.shape[0]]


setattr(_StaticDomainEstimator, "predict" + "_proba", _static_probabilities)


def _domain_features() -> tuple[np.ndarray, np.ndarray]:
    return np.asarray([[0.0], [1.0]], dtype=float), np.asarray([[2.0], [3.0]], dtype=float)


def test_domain_importance_rejects_probability_class_column_mismatch() -> None:
    source, target = _domain_features()
    estimator = _StaticDomainEstimator([[1.0]], classes=(0, 1))

    with pytest.raises(ValueError, match="classes_ length"):
        fit_domain_classifier_importance_weights(source, target, estimator=estimator)


def test_domain_importance_rejects_probability_missing_source_class() -> None:
    source, target = _domain_features()
    estimator = _StaticDomainEstimator([[1.0, 0.0]], classes=(1, 2))

    with pytest.raises(ValueError, match="source-domain label 0"):
        fit_domain_classifier_importance_weights(source, target, estimator=estimator)


def test_domain_importance_rejects_non_normalized_probability_rows() -> None:
    source, target = _domain_features()
    estimator = _StaticDomainEstimator([[0.8, 0.8]], classes=(0, 1))

    with pytest.raises(ValueError, match="rows must sum to 1"):
        fit_domain_classifier_importance_weights(source, target, estimator=estimator)
