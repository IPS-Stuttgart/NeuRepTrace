from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.target_prior_adjustment import adjust_target_probabilities_to_prior


def test_target_prior_adjustment_rejects_zero_mass_probability_rows() -> None:
    with pytest.raises(ValueError, match="probability rows must have positive mass"):
        adjust_target_probabilities_to_prior(
            [[0.0, 0.0], [0.7, 0.3]],
            config={"estimator": "mean"},
        )


def test_target_prior_adjustment_still_accepts_zero_class_probability() -> None:
    result = adjust_target_probabilities_to_prior(
        [[0.0, 1.0], [0.7, 0.3]],
        config={"estimator": "mean", "strength": 0.0},
    )

    assert np.all(np.isfinite(result.original_probabilities))
    np.testing.assert_allclose(result.original_probabilities.sum(axis=1), 1.0)
