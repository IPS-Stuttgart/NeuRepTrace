from __future__ import annotations

import numpy as np

from neureptrace.decoding.prior_shift import (
    adapt_probabilities_for_prior_shift,
    adapt_probability_blocks_for_prior_shift,
    reweight_probabilities_by_prior,
)


def _row_generator(rows: list[list[float]]):
    return ((float(value) for value in row) for row in rows)


def test_prior_shift_accepts_one_pass_probability_and_prior_iterables() -> None:
    result = adapt_probabilities_for_prior_shift(
        _row_generator([[0.6, 0.4], [0.2, 0.8]]),
        source_prior=(value for value in [0.5, 0.5]),
        target_prior=(value for value in [0.75, 0.25]),
    )

    assert result.probabilities.shape == (2, 2)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert np.allclose(result.target_prior, [0.75, 0.25])


def test_prior_shift_reweight_accepts_one_pass_probability_and_prior_iterables() -> None:
    adjusted = reweight_probabilities_by_prior(
        _row_generator([[0.5, 0.5]]),
        source_prior=(value for value in [0.5, 0.5]),
        target_prior=(value for value in [0.8, 0.2]),
    )

    assert np.allclose(adjusted, [[0.8, 0.2]])


def test_blockwise_prior_shift_reuses_materialized_one_pass_source_prior() -> None:
    result = adapt_probability_blocks_for_prior_shift(
        _row_generator([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]),
        (block for block in ["a", "a", "b", "b"]),
        source_prior=(value for value in [0.5, 0.5]),
        smoothing=0.0,
        min_block_rows=2,
    )

    assert set(result.block_results) == {"a", "b"}
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)


def test_blockwise_prior_shift_reuses_materialized_one_pass_target_priors() -> None:
    result = adapt_probability_blocks_for_prior_shift(
        _row_generator([[0.6, 0.4], [0.4, 0.6], [0.7, 0.3], [0.3, 0.7]]),
        (block for block in ["a", "a", "b", "b"]),
        source_prior=(value for value in [0.5, 0.5]),
        target_prior=(value for value in [0.8, 0.2]),
        min_block_rows=2,
    )

    assert set(result.block_results) == {"a", "b"}
    for block_result in result.block_results.values():
        assert block_result.n_iterations == 0
        assert np.allclose(block_result.target_prior, [0.8, 0.2])
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
