from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_free_target_prior import (
    apply_target_prior_correction,
    estimate_target_class_prior,
    fit_source_free_target_prior_predict_proba,
    fit_target_prior_corrected_source_model,
)


class _BiasedSourceModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probabilities = np.full((features.shape[0], 2), [0.90, 0.10], dtype=float)
        probabilities[features[:, 0] > 0.0] = [0.70, 0.30]
        return probabilities


class _PrototypeCollapseSourceModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probabilities = np.full((features.shape[0], 2), [0.92, 0.08], dtype=float)
        probabilities[features[:, 0] > 0.0] = [0.62, 0.38]
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
    assert corrected.metadata["source_free_target_prior_correction_stage"] == "post"
    assert corrected.metadata["source_free_uses_target_labels"] is False
    assert corrected.metadata["source_free_target_prior_uses_target_labels"] is False
    assert corrected.metadata["source_free_valid_for_benchmark"] is True
    assert corrected.probabilities[:, 1].mean() > uncorrected.probabilities[:, 1].mean()
    assert np.allclose(corrected.probabilities.sum(axis=1), 1.0)


def test_pre_adaptation_target_prior_correction_affects_pseudo_label_selection():
    target_features = np.vstack([np.full((6, 2), -1.0), np.full((6, 2), 1.0)])

    post_only = fit_source_free_target_prior_predict_proba(
        source_model=_PrototypeCollapseSourceModel(),
        target_features=target_features,
        confidence_threshold=0.80,
        max_iterations=2,
        min_class_count=2,
        min_active_classes=2,
        prototype_weight=0.5,
        target_prior_correction="balanced",
        target_prior_correction_stage="post",
    )
    pre = fit_source_free_target_prior_predict_proba(
        source_model=_PrototypeCollapseSourceModel(),
        target_features=target_features,
        confidence_threshold=0.80,
        max_iterations=2,
        min_class_count=2,
        min_active_classes=2,
        prototype_weight=0.5,
        target_prior_correction="balanced",
        target_prior_correction_stage="pre",
    )

    assert post_only.base_result.metadata["source_free_active_classes"] == 1
    assert pre.base_result.metadata["source_free_active_classes"] == 2
    assert pre.metadata["source_free_target_prior_correction_stage"] == "pre"
    assert pre.metadata["source_free_target_prior_uses_target_labels"] is False
    assert pre.probabilities[:, 1].mean() > post_only.probabilities[:, 1].mean()


def test_pre_adaptation_corrected_source_model_is_reusable():
    target_features = np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=float)
    wrapped = fit_target_prior_corrected_source_model(
        source_model=_BiasedSourceModel(),
        target_features=target_features,
        mode="balanced",
    )

    probabilities = wrapped.predict_proba(target_features)

    assert wrapped.classes_.tolist() == [0, 1]
    assert probabilities[1, 1] > _BiasedSourceModel().predict_proba(target_features)[1, 1]
    assert np.allclose(probabilities.sum(axis=1), 1.0)


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


@pytest.mark.parametrize("stage", ["during", "target", object()])
def test_target_prior_stage_rejects_invalid_values(stage):
    with pytest.raises(ValueError, match="target_prior_correction_stage"):
        fit_source_free_target_prior_predict_proba(
            source_model=_BiasedSourceModel(),
            target_features=np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=float),
            target_prior_correction_stage=stage,
        )
