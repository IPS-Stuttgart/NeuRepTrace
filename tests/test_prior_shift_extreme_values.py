from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.prior_shift import (
    adapt_probabilities_for_prior_shift,
    reweight_probabilities_by_prior,
)


def test_prior_shift_normalizes_extreme_finite_probability_rows() -> None:
    probabilities = np.asarray(
        [[1.0e308, 1.0e308], [1.0e308, 1.0e307]],
        dtype=float,
    )

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = adapt_probabilities_for_prior_shift(
            probabilities,
            source_prior=[0.5, 0.5],
            target_prior=[0.5, 0.5],
        )

    np.testing.assert_allclose(
        result.probabilities,
        [[0.5, 0.5], [10.0 / 11.0, 1.0 / 11.0]],
    )
    np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0)
    assert np.all(np.isfinite(result.probabilities))


def test_prior_shift_normalizes_extreme_finite_prior_vectors() -> None:
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        adjusted = reweight_probabilities_by_prior(
            [[0.5, 0.5]],
            source_prior=[1.0e308, 1.0e308],
            target_prior=[1.0e308, 1.0e307],
        )

    np.testing.assert_allclose(adjusted, [[10.0 / 11.0, 1.0 / 11.0]])
    assert np.all(np.isfinite(adjusted))


def test_prior_shift_preserves_positive_mass_floor_after_scaling() -> None:
    with pytest.raises(ValueError, match="positive mass"):
        adapt_probabilities_for_prior_shift(
            [[1.0e-13, 1.0e-13]],
            epsilon=1.0e-12,
        )

    with pytest.raises(ValueError, match="positive mass"):
        reweight_probabilities_by_prior(
            [[0.5, 0.5]],
            source_prior=[1.0e-13, 1.0e-13],
            target_prior=[0.5, 0.5],
            epsilon=1.0e-12,
        )
