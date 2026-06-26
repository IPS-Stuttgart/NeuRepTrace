from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_free_target_prior import (
    apply_target_prior_correction,
    estimate_target_class_prior,
    fit_source_free_target_prior_predict_proba,
    stabilize_target_class_prior,
)


class _BiasedSourceModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probabilities = np.full((features.shape[0], 2), [0.90, 0.10], dtype=float)
        probabilities[features[:, 0] > 0.0] = [0.70, 0.30]
        return probabilities


class _ImbalancedPseudoLabelSourceModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probabilities = np.tile(np.array([[0.94, 0.06]], dtype=float), (features.shape[0], 1))
        probabilities[features[:, 0] > 0.0] = np.array([0.42, 0.58], dtype=float)
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


def test_target_prior_wrapper_forwards_balanced_topk_selection():
    target_features = np.vstack([np.full((10, 2), -1.0), np.full((4, 2), 1.0)])

    result = fit_source_free_target_prior_predict_proba(
        source_model=_ImbalancedPseudoLabelSourceModel(),
        target_features=target_features,
        confidence_threshold=0.80,
        max_iterations=2,
        min_class_count=2,
        min_active_classes=2,
        prototype_weight=0.5,
        pseudo_label_selection="balanced_topk",
        balanced_topk_per_class=2,
        target_prior_correction="none",
    )

    metadata = result.metadata
    assert metadata["source_free_pseudo_label_selection"] == "balanced_topk"
    assert metadata["source_free_balanced_topk_per_class"] == 2
    assert metadata["source_free_active_classes"] == 2
    assert result.base_result.adapter.prototype_class_counts_.tolist() == [2, 2]
    assert result.base_result.adapter.selected_.sum() == 4


def test_target_prior_strength_interpolates_correction():
    probabilities = np.array([[0.90, 0.10], [0.70, 0.30]], dtype=float)

    none, prior = apply_target_prior_correction(probabilities, mode="balanced", strength=0.0)
    partial, _ = apply_target_prior_correction(probabilities, mode="balanced", strength=0.5, prior=prior)
    full, _ = apply_target_prior_correction(probabilities, mode="balanced", strength=1.0, prior=prior)

    assert np.allclose(none, probabilities)
    assert none[:, 1].mean() < partial[:, 1].mean() < full[:, 1].mean()


def test_disabled_target_prior_correction_ignores_irrelevant_prior_argument():
    probabilities = np.array([[0.90, 0.10], [0.70, 0.30]], dtype=float)

    corrected, prior = apply_target_prior_correction(
        probabilities,
        mode="none",
        prior=np.array([1.0]),
    )

    assert np.allclose(corrected, probabilities)
    assert np.allclose(prior, [0.8, 0.2])


def test_smoothed_target_prior_makes_balancing_less_aggressive():
    probabilities = np.array([[0.98, 0.02], [0.92, 0.08]], dtype=float)

    raw, _ = apply_target_prior_correction(probabilities, mode="balanced", strength=0.0)
    smoothed, smoothed_prior = apply_target_prior_correction(
        probabilities,
        mode="balanced",
        strength=1.0,
        smoothing=0.5,
    )
    full, full_prior = apply_target_prior_correction(
        probabilities,
        mode="balanced",
        strength=1.0,
        smoothing=0.0,
    )

    assert full_prior[1] < smoothed_prior[1] < 0.5
    assert raw[:, 1].mean() < smoothed[:, 1].mean() < full[:, 1].mean()


def test_confidence_weighted_target_prior_downweights_ambiguous_rows():
    probabilities = np.array(
        [
            [0.99, 0.01],
            [0.50, 0.50],
            [0.51, 0.49],
        ],
        dtype=float,
    )

    mean_prior = estimate_target_class_prior(probabilities, estimator="mean")
    confidence_prior = estimate_target_class_prior(probabilities, estimator="confidence_weighted")
    entropy_prior = estimate_target_class_prior(probabilities, estimator="entropy_weighted")

    assert confidence_prior[0] > mean_prior[0]
    assert entropy_prior[0] > mean_prior[0]
    assert np.allclose(confidence_prior.sum(), 1.0)
    assert np.allclose(entropy_prior.sum(), 1.0)


def test_target_prior_metadata_records_raw_and_stabilized_prior():
    target_features = np.array([[-2.0, 0.0], [-1.0, 0.2], [0.25, 0.0], [1.0, 0.1]], dtype=float)

    result = fit_source_free_target_prior_predict_proba(
        source_model=_BiasedSourceModel(),
        target_features=target_features,
        max_iterations=0,
        target_prior_correction="balanced",
        target_prior_estimator="confidence_weighted",
        target_prior_smoothing=0.25,
        target_prior_floor=0.05,
    )

    metadata = result.metadata
    assert metadata["source_free_target_prior_estimator"] == "confidence_weighted"
    assert metadata["source_free_target_prior_smoothing"] == 0.25
    assert metadata["source_free_target_prior_floor"] == 0.05
    assert metadata["source_free_target_raw_class_prior"] != metadata["source_free_target_class_prior"]
    assert result.probabilities.shape == (4, 2)


def test_target_class_prior_is_normalized_and_validated():
    prior = estimate_target_class_prior(np.array([[2.0, 0.0], [1.0, 1.0]], dtype=float))

    assert np.allclose(prior.sum(), 1.0)
    assert np.all(prior > 0.0)


def test_target_prior_floor_stabilizes_extreme_priors():
    prior = stabilize_target_class_prior(np.array([0.99, 0.01]), floor=0.10)

    assert np.allclose(prior.sum(), 1.0)
    assert prior[1] > 0.01


@pytest.mark.parametrize("strength", [-0.1, 1.1, True, np.nan])
def test_target_prior_strength_rejects_invalid_values(strength):
    with pytest.raises(ValueError, match="target_prior_strength"):
        apply_target_prior_correction(np.array([[0.8, 0.2]], dtype=float), strength=strength)


@pytest.mark.parametrize("smoothing", [-0.1, 1.1, True, np.nan])
def test_target_prior_smoothing_rejects_invalid_values(smoothing):
    with pytest.raises(ValueError, match="target_prior_smoothing"):
        apply_target_prior_correction(np.array([[0.8, 0.2]], dtype=float), smoothing=smoothing)


@pytest.mark.parametrize("floor", [-0.1, 1.0, True, np.nan])
def test_target_prior_floor_rejects_invalid_values(floor):
    with pytest.raises(ValueError, match="target_prior_floor"):
        apply_target_prior_correction(np.array([[0.8, 0.2]], dtype=float), floor=floor)


def test_target_prior_floor_rejects_impossible_floor():
    with pytest.raises(ValueError, match="target_prior_floor.*1 / n_classes"):
        apply_target_prior_correction(np.array([[0.8, 0.2]], dtype=float), floor=0.5)
