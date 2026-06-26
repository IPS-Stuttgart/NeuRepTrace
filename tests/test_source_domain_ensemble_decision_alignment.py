from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator

from neureptrace.decoding.source_ensemble import fit_source_domain_probability_ensemble


class BinaryDecisionEstimator(BaseEstimator):
    """Estimator with decision_function but no predict_proba.

    The fitted domain may contain a strict subset of global classes.  Scikit-learn
    uses classes_[0] for negative binary scores and classes_[1] for positive
    scores, so the ensemble alignment must route those two columns by classes_,
    not by the first two global columns.
    """

    def fit(self, X, y):  # noqa: N803 - sklearn-compatible signature
        del X
        self.classes_ = np.asarray(tuple(dict.fromkeys(np.asarray(y, dtype=object).tolist())), dtype=object)
        return self

    def decision_function(self, X):  # noqa: N803 - sklearn-compatible signature
        return np.asarray(X, dtype=float)[:, 0]


def test_binary_decision_function_alignment_uses_model_classes_for_subset_domains() -> None:
    source_features = np.asarray(
        [
            [-4.0],
            [-3.0],
            [-2.0],
            [2.0],
            [3.0],
            [4.0],
        ],
        dtype=float,
    )
    source_labels = np.asarray([0, 0, 1, 2, 2, 2], dtype=object)
    source_domains = np.asarray(["skipped", "skipped", "decision", "decision", "decision", "decision"], dtype=object)
    target_features = np.asarray([[-1.0], [1.0]], dtype=float)

    result = fit_source_domain_probability_ensemble(
        source_features=source_features,
        source_labels=source_labels,
        source_domains=source_domains,
        target_features=target_features,
        estimator=BinaryDecisionEstimator(),
        min_classes_per_domain=2,
    )

    assert set(result.models) == {"decision"}
    assert result.classes.tolist() == [0, 1, 2]
    assert result.predictions.tolist() == [1, 2]
    assert result.probabilities[0, 1] > result.probabilities[0, 0]
    assert result.probabilities[0, 1] > result.probabilities[0, 2]
    assert result.probabilities[1, 2] > result.probabilities[1, 0]
    assert result.probabilities[1, 2] > result.probabilities[1, 1]
