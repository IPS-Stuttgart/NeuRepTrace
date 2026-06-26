from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_free_target_prior import fit_source_free_target_prior_predict_proba


class _CollapsedPseudoLabelSourceModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probabilities = np.tile(np.array([[0.64, 0.36]], dtype=float), (features.shape[0], 1))
        probabilities[features[:, 0] > 0.0] = np.array([0.56, 0.44], dtype=float)
        return probabilities


def test_target_prior_wrapper_forwards_source_free_prototype_estimator():
    target_features = np.vstack([np.full((8, 2), -1.0), np.full((8, 2), 1.0)])

    result = fit_source_free_target_prior_predict_proba(
        source_model=_CollapsedPseudoLabelSourceModel(),
        target_features=target_features,
        confidence_threshold=0.90,
        max_iterations=2,
        min_class_count=2,
        min_active_classes=2,
        prototype_weight=0.5,
        prototype_estimator="soft_all",
        target_prior_correction="none",
    )

    assert result.metadata["source_free_prototype_estimator"] == "soft_all"
    assert result.metadata["source_free_active_classes"] == 2
    assert result.metadata["source_free_target_prior_correction"] == "none"
    assert result.metadata["source_free_target_prior_uses_target_labels"] is False
    assert result.probabilities.shape == (target_features.shape[0], 2)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
