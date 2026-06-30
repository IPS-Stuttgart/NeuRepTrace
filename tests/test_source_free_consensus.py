from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_free_consensus import (
    SourceFreeConsensusVariant,
    combine_probability_variants,
    estimate_consensus_variant_weights,
    fit_source_free_consensus_predict_proba,
)


class _ToySourceModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probabilities = np.full((features.shape[0], 2), [0.90, 0.10], dtype=float)
        probabilities[features[:, 0] > 0.0] = [0.35, 0.65]
        return probabilities


def test_combine_probability_variants_logit_mean_normalizes_rows() -> None:
    first = np.array([[0.80, 0.20], [0.40, 0.60]], dtype=float)
    second = np.array([[0.60, 0.40], [0.20, 0.80]], dtype=float)

    combined = combine_probability_variants([first, second], weights=[0.25, 0.75], mode="logit_mean")

    assert combined.shape == first.shape
    assert np.allclose(combined.sum(axis=1), 1.0)
    assert combined[0, 0] > combined[0, 1]
    assert combined[1, 1] > combined[1, 0]


def test_consensus_weights_can_penalize_collapsed_confident_variant() -> None:
    collapsed = np.tile(np.array([[0.99, 0.01]], dtype=float), (6, 1))
    balanced = np.array(
        [
            [0.75, 0.25],
            [0.70, 0.30],
            [0.65, 0.35],
            [0.35, 0.65],
            [0.30, 0.70],
            [0.25, 0.75],
        ],
        dtype=float,
    )

    weights = estimate_consensus_variant_weights(
        [collapsed, balanced],
        confidence_weight=0.5,
        balance_weight=2.0,
        temperature=0.5,
    )

    assert np.allclose(weights.sum(), 1.0)
    assert weights[1] > weights[0]


def test_fit_source_free_consensus_returns_protocol_metadata() -> None:
    target_features = np.array(
        [
            [-1.0, 0.0],
            [-0.5, 0.1],
            [0.25, -0.1],
            [0.75, 0.2],
        ],
        dtype=float,
    )

    result = fit_source_free_consensus_predict_proba(
        source_model=_ToySourceModel(),
        target_features=target_features,
        variants=[
            SourceFreeConsensusVariant("raw", {"max_iterations": 0, "target_prior_correction": "none"}),
            SourceFreeConsensusVariant(
                "prior",
                {
                    "max_iterations": 0,
                    "target_prior_correction": "balanced",
                    "target_prior_strength": 0.5,
                    "target_prior_smoothing": 0.25,
                },
            ),
        ],
    )

    assert result.probabilities.shape == (4, 2)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert result.weights.shape == (2,)
    assert np.allclose(result.weights.sum(), 1.0)
    assert result.metadata["source_free_consensus"] is True
    assert result.metadata["source_free_consensus_uses_target_labels"] is False
    assert result.metadata["source_free_consensus_uses_source_rows_during_adaptation"] is False
    assert result.metadata["source_free_consensus_valid_for_protocol_2_5"] is True
    assert result.metadata["source_free_consensus_variants"] == "raw|prior"


def test_fit_source_free_consensus_accepts_named_default_variants() -> None:
    target_features = np.array([[-1.0], [0.5], [1.0]], dtype=float)

    result = fit_source_free_consensus_predict_proba(
        source_model=_ToySourceModel(),
        target_features=target_features,
        variants=["source_raw", "robust_prior"],
    )

    assert result.probabilities.shape == (3, 2)
    assert result.metadata["source_free_consensus_n_variants"] == 2


def test_consensus_rejects_mixed_fixed_and_adaptive_weights() -> None:
    with pytest.raises(ValueError, match="Either specify all"):
        fit_source_free_consensus_predict_proba(
            source_model=_ToySourceModel(),
            target_features=np.array([[-1.0], [1.0]], dtype=float),
            variants=[
                SourceFreeConsensusVariant("raw", {"max_iterations": 0}, weight=0.5),
                SourceFreeConsensusVariant("prior", {"max_iterations": 0}),
            ],
        )


def test_combine_probability_variants_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        combine_probability_variants(
            [
                np.array([[0.5, 0.5]], dtype=float),
                np.array([[0.5, 0.5], [0.6, 0.4]], dtype=float),
            ]
        )


def test_combine_probability_variants_rejects_boolean_probabilities() -> None:
    boolean_probabilities = np.array([[True, False], [False, True]])
    numeric_probabilities = np.array([[0.70, 0.30], [0.20, 0.80]], dtype=float)

    with pytest.raises(ValueError, match="probabilities must be numeric, not boolean"):
        combine_probability_variants([boolean_probabilities, numeric_probabilities])


def test_combine_probability_variants_rejects_boolean_weights() -> None:
    first = np.array([[0.80, 0.20], [0.40, 0.60]], dtype=float)
    second = np.array([[0.60, 0.40], [0.20, 0.80]], dtype=float)

    with pytest.raises(ValueError, match="weights must be numeric, not boolean"):
        combine_probability_variants([first, second], weights=[True, False])


def test_fit_source_free_consensus_rejects_boolean_fixed_variant_weights() -> None:
    with pytest.raises(ValueError, match="weights must be numeric, not boolean"):
        fit_source_free_consensus_predict_proba(
            source_model=_ToySourceModel(),
            target_features=np.array([[-1.0], [1.0]], dtype=float),
            variants=[
                SourceFreeConsensusVariant("raw", {"max_iterations": 0, "target_prior_correction": "none"}, weight=True),
                SourceFreeConsensusVariant("prior", {"max_iterations": 0, "target_prior_correction": "none"}, weight=0.5),
            ],
        )
