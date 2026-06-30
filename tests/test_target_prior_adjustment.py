from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.target_prior_adjustment import (
    TARGET_PRIOR_ADJUSTMENT_CATEGORY,
    adjust_target_probabilities_to_prior,
    estimate_target_prior_em,
    estimate_target_prior_mean,
    normalize_prior_estimator,
    target_prior_adjustment_config,
)


def test_mean_prior_adjustment_metadata_and_shapes() -> None:
    probabilities = np.asarray([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8]], dtype=float)

    result = adjust_target_probabilities_to_prior(probabilities, config={"estimator": "mean"})

    assert result.probabilities.shape == probabilities.shape
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert np.allclose(result.target_prior, np.asarray([probabilities[:, 0].mean(), probabilities[:, 1].mean()]))
    assert result.metadata["target_prior_adjustment_protocol_category"] == TARGET_PRIOR_ADJUSTMENT_CATEGORY
    assert result.metadata["target_prior_adjustment_uses_target_labels"] is False
    assert result.metadata["target_prior_adjustment_valid_for_unlabeled_target_adaptation"] is True


def test_em_prior_estimate_is_valid_distribution() -> None:
    probabilities = np.asarray([[0.95, 0.05], [0.9, 0.1], [0.85, 0.15], [0.1, 0.9]], dtype=float)

    prior, n_iter, converged = estimate_target_prior_em(probabilities, source_prior=[0.5, 0.5], max_iter=100, tol=1e-7)

    assert prior.shape == (2,)
    assert np.isclose(np.sum(prior), 1.0)
    assert n_iter >= 1
    assert isinstance(converged, bool)
    assert prior[0] > prior[1]


def test_strength_zero_returns_original_normalized_probabilities() -> None:
    probabilities = np.asarray([[2.0, 1.0], [1.0, 3.0]], dtype=float)

    result = adjust_target_probabilities_to_prior(probabilities, config={"strength": 0.0, "estimator": "mean"})

    assert np.allclose(result.probabilities, result.original_probabilities)


def test_prior_aliases_and_validation() -> None:
    assert normalize_prior_estimator("average") == "mean"
    assert normalize_prior_estimator("expectation-maximization") == "em"
    cfg = target_prior_adjustment_config(strength="0.25", max_iter="3")
    assert cfg.strength == 0.25
    assert cfg.max_iter == 3

    with pytest.raises(ValueError, match="estimator"):
        normalize_prior_estimator("bad")
    with pytest.raises(ValueError, match="strength"):
        target_prior_adjustment_config(strength=1.5)


def test_probability_validation() -> None:
    with pytest.raises(ValueError, match="at least two classes"):
        estimate_target_prior_mean([[1.0], [1.0]])

    with pytest.raises(ValueError, match="source_prior"):
        adjust_target_probabilities_to_prior([[0.4, 0.6]], config={"source_prior": [1.0, 0.0, 0.0]})
