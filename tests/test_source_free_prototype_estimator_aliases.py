from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_free import fit_source_free_predict_proba


class _CollapsedPseudoLabelSourceModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probabilities = np.tile(np.array([[0.55, 0.45]], dtype=float), (features.shape[0], 1))
        probabilities[features[:, 0] > 0.0] = np.array([0.51, 0.49], dtype=float)
        return probabilities


@pytest.mark.parametrize("prototype_estimator", ["", "argmax"])
def test_source_free_hard_prototype_estimator_aliases_normalize_to_hard(prototype_estimator: str) -> None:
    target_features = np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=float)

    result = fit_source_free_predict_proba(
        source_model=_CollapsedPseudoLabelSourceModel(),
        target_features=target_features,
        max_iterations=0,
        prototype_estimator=prototype_estimator,
    )

    assert result.metadata["source_free_prototype_estimator"] == "hard"
    assert result.probabilities.shape == (2, 2)
