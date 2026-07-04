from __future__ import annotations

import numpy as np

import neureptrace._source_free_soft_prototypes_patch as soft_prototypes_patch
import neureptrace.decoding.source_free_target_prior as target_prior
from neureptrace.decoding import source_free


def test_soft_prototype_patch_refreshes_loaded_target_prior_dispatch():
    def stale_fit_source_free_predict_proba(**_kwargs):  # pragma: no cover - must be replaced by install()
        raise AssertionError("target-prior wrapper kept a stale source-free dispatch")

    target_prior.fit_source_free_predict_proba = stale_fit_source_free_predict_proba
    soft_prototypes_patch.install()

    assert target_prior.fit_source_free_predict_proba is source_free.fit_source_free_predict_proba

    class _CollapsedPseudoLabelSourceModel:
        classes_ = np.array([0, 1])

        def predict_proba(self, features: np.ndarray) -> np.ndarray:
            probabilities = np.tile(np.array([[0.64, 0.36]], dtype=float), (features.shape[0], 1))
            probabilities[features[:, 0] > 0.0] = np.array([0.56, 0.44], dtype=float)
            return probabilities

    target_features = np.vstack([np.full((8, 2), -1.0), np.full((8, 2), 1.0)])
    result = target_prior.fit_source_free_target_prior_predict_proba(
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
    assert result.probabilities.shape == (target_features.shape[0], 2)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
