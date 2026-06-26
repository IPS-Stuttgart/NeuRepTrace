from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_free_target_prior import (
    apply_target_prior_correction,
    estimate_target_class_prior,
    fit_source_free_target_prior_predict_proba,
)


class _BiasedSourceModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probabilities = np.full((features.shape[0], 2), [0.90, 0.10], dtype=float)
        probabilities[features[:, 0] > 0.0] = [0.70, 0.30]
        return probabilities


def test_balanced_target_prior_correction_uses_unlabeled_target_predictions():
    target_features = np.array(
        [
            [-2.0, 0.0],
            [-1.0, 0.2],
            [-0.5, -0.1],
            [0.25, 0.0],
            [1.0, 0.1],
            [1.5, -0.1],
        ],
        dtype=float,
    )
    source_model = _BiasedSourceModel()

    uncorrected = fit_source_free_target_prior_predict_proba(
        source_model=source_model,
        target_features=target_features,
        max_iterations=0,
        target_prior_correction="none",
    )
    corrected = fit_source_free_target_prior_predict_proba(
        source_model=source_model,
        target_features=target_features,
        max_iterations=0,
        target_prior_correction="balanced",
    )

    assert corrected.metadata["source_free_target_prior_correction"] == "balanced"
    assert corrected.metadata["source_free_uses_target_labels"] is False
    assert corrected.metadata["source_free_target_prior_uses_target_labels"] is False
    assert corrected.metadata["source_free_valid_for_benchmark"] is True
    assert corrected.probabilities[:, 1].mean() > uncorrected.probabilities[:, 1].mean()
    assert np.allclose(corrected.probabilities.sum(axis=1), 1.0)


def test_target_prior_strength_interpolates_correction():
    probabilities = np.array([[0.90, 0.10], [0.70, 0.30]], dtype=float)

    none, prior = apply_target_prior_correction(probabilities, mode="balanced", strength=0.0)
    partial, _ = apply_target_prior_correction(probabilities, mode="balanced", strength=0.5, prior=prior)
    full, _ = apply_target_prior_correction(probabilities, mode="balanced", strength=1.0, prior=prior)

    assert np.allclose(none, probabilities)
    assert none[:, 1].mean() < partial[:, 1].mean() < full[:, 1].mean()


def test_target_class_prior_is_normalized_and_validated():
    prior = estimate_target_class_prior(np.array([[2.0, 0.0], [1.0, 1.0]], dtype=float))

    assert np.allclose(prior.sum(), 1.0)
    assert np.all(prior > 0.0)


@pytest.mark.parametrize("strength", [-0.1, 1.1, True, np.nan])
def test_target_prior_strength_rejects_invalid_values(strength):
    with pytest.raises(ValueError, match="target_prior_strength"):
        apply_target_prior_correction(np.array([[0.8, 0.2]], dtype=float), strength=strength)
