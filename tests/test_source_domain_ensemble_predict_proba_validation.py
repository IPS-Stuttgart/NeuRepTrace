from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from neureptrace.decoding.source_ensemble import fit_source_domain_probability_ensemble


class RawProbabilityEstimator(BaseEstimator):
    """Estimator returning a configured raw probability output."""

    def __init__(self, output=((0.5, 0.5), (0.5, 0.5))):
        self.output = output

    def fit(self, features, labels):
        self.classes_ = np.unique(labels)
        return self

    def predict_proba(self, features):
        return np.asarray(self.output, dtype=float)


def _fit_with_probability_output(output):
    return fit_source_domain_probability_ensemble(
        source_features=[[-2.0], [-1.5], [1.5], [2.0]],
        source_labels=[0, 0, 1, 1],
        source_domains=["source", "source", "source", "source"],
        target_features=[[-1.7], [1.7]],
        estimator=RawProbabilityEstimator(output=output),
    )


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (((1.2, -0.2), (1.2, -0.2)), "between 0 and 1"),
        (((0.2, 0.2), (0.2, 0.2)), "sum to 1"),
        (((np.nan, np.nan), (np.nan, np.nan)), "finite values"),
        ((0.5, 0.5), "two-dimensional matrix"),
        (((0.5, 0.5),), "one row per feature row"),
    ],
)
def test_source_domain_ensemble_rejects_malformed_predict_proba_output(output, message) -> None:
    with pytest.raises(ValueError, match=message):
        _fit_with_probability_output(output)


def test_source_domain_ensemble_accepts_tiny_predict_proba_roundoff() -> None:
    result = _fit_with_probability_output(
        (
            (1.0 + 5e-7, -5e-7),
            (1.0 + 5e-7, -5e-7),
        )
    )

    np.testing.assert_allclose(result.probabilities.sum(axis=1), np.ones(2))
    np.testing.assert_allclose(result.probabilities[:, 0], np.ones(2), atol=1e-6)
