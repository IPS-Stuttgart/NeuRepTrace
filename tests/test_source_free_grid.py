from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_free_grid import fit_source_free_grid_predict_proba, score_probability_shape


class _BiasedSourceModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probabilities = np.tile(np.array([[0.90, 0.10]], dtype=float), (features.shape[0], 1))
        probabilities[features[:, 0] > 0.0] = np.array([0.70, 0.30], dtype=float)
        return probabilities


def test_source_free_grid_selects_unlabeled_prior_strength_for_biased_predictions():
    target_features = np.vstack([np.full((6, 2), -1.0), np.full((6, 2), 1.0)])

    result = fit_source_free_grid_predict_proba(
        source_model=_BiasedSourceModel(),
        target_features=target_features,
        max_iterations=0,
        prototype_weights=(0.0,),
        confidence_thresholds=(0.75,),
        prior_strengths=(0.0, 1.0),
        pseudo_label_selections=("confidence",),
    )

    assert result.metadata["source_free_grid_selection"] is True
    assert result.metadata["source_free_grid_prior_strength"] == 1.0
    assert result.metadata["source_free_grid_candidate_count"] == 2
    assert result.probabilities[:, 1].mean() > _BiasedSourceModel().predict_proba(target_features)[:, 1].mean()
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)


def test_source_free_grid_accepts_scalar_grid_options_without_splitting_strings():
    target_features = np.vstack([np.full((4, 2), -1.0), np.full((4, 2), 1.0)])

    result = fit_source_free_grid_predict_proba(
        source_model=_BiasedSourceModel(),
        target_features=target_features,
        max_iterations=0,
        prototype_weights=np.float64(0.0),
        confidence_thresholds=0.75,
        prior_strengths=np.asarray(0.0),
        pseudo_label_selections="confidence",
        balanced_topk_per_class_values=None,
    )

    assert result.metadata["source_free_grid_selection"] is True
    assert result.metadata["source_free_grid_candidate_count"] == 1
    assert result.ranked[0]["selection"] == "confidence"
    assert result.ranked[0]["topk"] is None
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)


def test_source_free_grid_rejects_empty_grid_options():
    target_features = np.vstack([np.full((2, 2), -1.0), np.full((2, 2), 1.0)])

    with pytest.raises(ValueError, match="confidence_thresholds"):
        fit_source_free_grid_predict_proba(
            source_model=_BiasedSourceModel(),
            target_features=target_features,
            max_iterations=0,
            confidence_thresholds=(),
            prototype_weights=(0.0,),
            prior_strengths=(0.0,),
            pseudo_label_selections=("confidence",),
        )


def test_source_free_grid_can_rank_balanced_topk_variants():
    target_features = np.vstack([np.full((5, 2), -1.0), np.full((3, 2), 1.0)])

    result = fit_source_free_grid_predict_proba(
        source_model=_BiasedSourceModel(),
        target_features=target_features,
        confidence_thresholds=(0.8,),
        prototype_weights=(0.5,),
        prior_strengths=(0.0,),
        pseudo_label_selections=("confidence", "balanced_topk"),
        balanced_topk_per_class_values=(None, 2),
        min_class_count=1,
    )

    assert len(result.ranked) >= 2
    assert result.metadata["source_free_grid_candidate_count"] == len(result.ranked)
    assert result.metadata["source_free_uses_target_labels"] is False


@pytest.mark.parametrize(
    "prior_strength",
    [True, np.bool_(True), np.asarray(True), np.array([False]), np.array([0.5])],
)
def test_source_free_grid_rejects_invalid_prior_strength_scalars(prior_strength):
    target_features = np.vstack([np.full((2, 2), -1.0), np.full((2, 2), 1.0)])

    with pytest.raises(ValueError, match="prior_strength"):
        fit_source_free_grid_predict_proba(
            source_model=_BiasedSourceModel(),
            target_features=target_features,
            max_iterations=0,
            prototype_weights=(0.0,),
            confidence_thresholds=(0.75,),
            prior_strengths=(prior_strength,),
            pseudo_label_selections=("confidence",),
        )


def test_probability_shape_score_prefers_active_classes_when_other_terms_match():
    probabilities = np.array([[0.8, 0.2], [0.2, 0.8]], dtype=float)

    inactive_score, _ = score_probability_shape(probabilities, active_classes=0)
    active_score, terms = score_probability_shape(probabilities, active_classes=2)

    assert terms["active_fraction"] == 1.0
    assert active_score > inactive_score


def test_probability_shape_score_preserves_large_finite_ratios_under_strict_overflow():
    probabilities = np.array([[1e308, 1e308], [1e308, 1.0]], dtype=float)
    scaled = np.array([[1.0, 1.0], [1.0, 1e-308]], dtype=float)

    with np.errstate(over="raise", invalid="raise"):
        score, terms = score_probability_shape(probabilities, active_classes=2)

    expected_score, expected_terms = score_probability_shape(scaled, active_classes=2)
    assert score == pytest.approx(expected_score)
    for key, expected in expected_terms.items():
        assert terms[key] == pytest.approx(expected)
