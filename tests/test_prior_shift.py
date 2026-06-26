from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.prior_shift import (
    PRIOR_SHIFT_CATEGORY,
    adapt_probabilities_for_prior_shift,
    adapt_probability_blocks_for_prior_shift,
    prior_from_labels,
    reweight_probabilities_by_prior,
)


def test_fixed_target_prior_reweights_probability_rows() -> None:
    probabilities = np.asarray([[0.5, 0.5], [0.8, 0.2], [0.2, 0.8]], dtype=float)

    result = adapt_probabilities_for_prior_shift(
        probabilities,
        source_prior=[0.5, 0.5],
        target_prior=[0.75, 0.25],
    )

    assert result.n_iterations == 0
    assert result.converged is True
    assert np.allclose(result.target_prior, [0.75, 0.25])
    assert result.probabilities[0, 0] > probabilities[0, 0]
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert result.metadata["prior_shift_protocol_category"] == PRIOR_SHIFT_CATEGORY
    assert result.metadata["prior_shift_uses_target_probabilities"] is True
    assert result.metadata["prior_shift_uses_target_labels"] is False
    assert result.metadata["prior_shift_valid_for_strict_source_only"] is False
    assert result.metadata["prior_shift_valid_for_unlabeled_target_adaptation"] is True


def test_em_estimates_target_prior_from_unlabeled_probability_rows() -> None:
    class0_rows = np.tile(np.asarray([[0.95, 0.05]]), (8, 1))
    class1_rows = np.tile(np.asarray([[0.10, 0.90]]), (2, 1))
    probabilities = np.vstack([class0_rows, class1_rows])

    result = adapt_probabilities_for_prior_shift(
        probabilities,
        source_prior=[0.5, 0.5],
        max_iter=100,
        tol=1e-10,
        smoothing=0.0,
    )

    assert result.converged is True
    assert result.n_iterations > 0
    assert result.target_prior[0] > 0.75
    assert result.target_prior[1] < 0.25
    assert result.class_bias[0] > result.class_bias[1]
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)


def test_reweight_probabilities_by_prior_matches_expected_ratio() -> None:
    probabilities = np.asarray([[0.5, 0.5]], dtype=float)

    adjusted = reweight_probabilities_by_prior(probabilities, source_prior=[0.5, 0.5], target_prior=[0.8, 0.2])

    assert np.allclose(adjusted, [[0.8, 0.2]])


def test_blockwise_prior_shift_estimates_different_block_priors() -> None:
    block_a = np.tile(np.asarray([[0.9, 0.1]]), (5, 1))
    block_b = np.tile(np.asarray([[0.1, 0.9]]), (5, 1))
    probabilities = np.vstack([block_a, block_b])
    blocks = np.asarray(["a"] * 5 + ["b"] * 5, dtype=object)

    result = adapt_probability_blocks_for_prior_shift(
        probabilities,
        blocks,
        source_prior=[0.5, 0.5],
        smoothing=0.0,
    )

    assert set(result.block_results) == {"a", "b"}
    assert result.block_results["a"].target_prior[0] > 0.9
    assert result.block_results["b"].target_prior[1] > 0.9
    assert result.metadata["prior_shift_blockwise"] is True
    assert result.metadata["prior_shift_uses_target_labels"] is False
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)


def test_blockwise_prior_shift_rejects_tiny_blocks() -> None:
    probabilities = np.asarray([[0.6, 0.4], [0.4, 0.6]])

    with pytest.raises(ValueError, match="fewer than min_block_rows"):
        adapt_probability_blocks_for_prior_shift(probabilities, ["a", "b"], min_block_rows=2)


def test_prior_from_labels_uses_source_labels_only() -> None:
    prior, classes = prior_from_labels(["left", "left", "right", "left"], classes=["left", "right"])

    assert classes == ("left", "right")
    assert np.allclose(prior, [0.75, 0.25])


def test_prior_shift_rejects_bad_shapes_and_target_labels_api() -> None:
    with pytest.raises(ValueError, match="at least two classes"):
        adapt_probabilities_for_prior_shift([[1.0]])

    with pytest.raises(TypeError):
        adapt_probabilities_for_prior_shift(
            [[0.5, 0.5]],
            target_labels=[0],  # type: ignore[call-arg]
        )
