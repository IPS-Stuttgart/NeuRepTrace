from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from neureptrace.decoding.source_ensemble import fit_source_domain_probability_ensemble


class FixedProbabilityEstimator(BaseEstimator):
    """Estimator returning the same configured probability row for every sample."""

    def __init__(self, row=(0.5, 0.5)):
        self.row = row

    def fit(self, features, labels):
        self.classes_ = np.unique(labels)
        return self

    def predict_proba(self, features):
        row = np.asarray(self.row, dtype=float)
        return np.tile(row, (np.asarray(features).shape[0], 1))


def _fit_with_probability_row(row):
    return fit_source_domain_probability_ensemble(
        source_features=[[-2.0], [-1.5], [1.5], [2.0]],
        source_labels=[0, 0, 1, 1],
        source_domains=["source", "source", "source", "source"],
        target_features=[[-1.7], [1.7]],
        estimator=FixedProbabilityEstimator(row=row),
    )


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ((1.2, -0.2), "between 0 and 1"),
        ((0.2, 0.2), "sum to 1"),
        ((np.nan, np.nan), "finite values"),
    ],
)
def test_source_domain_ensemble_rejects_malformed_predict_proba_rows(row, message) -> None:
    with pytest.raises(ValueError, match=message):
        _fit_with_probability_row(row)


def test_source_domain_ensemble_accepts_tiny_predict_proba_roundoff() -> None:
    result = _fit_with_probability_row((1.0 + 5e-7, -5e-7))

    np.testing.assert_allclose(result.probabilities.sum(axis=1), np.ones(2))
    np.testing.assert_allclose(result.probabilities[:, 0], np.ones(2), atol=1e-6)
