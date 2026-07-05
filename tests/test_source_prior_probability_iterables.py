from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_prior import adjust_probabilities_to_source_prior


def test_accepts_one_pass_probability_iterables() -> None:
    probabilities = ((probability for probability in row) for row in ([0.75, 0.25], [0.50, 0.50]))

    result = adjust_probabilities_to_source_prior(
        probabilities,
        source_labels=["major", "major", "major", "minor"],
        classes=["major", "minor"],
        config={"target_prior": "uniform"},
    )

    assert result.probabilities.shape == (2, 2)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert result.probabilities[0, 1] > 0.25


def test_rejects_boolean_one_pass_probability_iterables() -> None:
    probabilities = ((probability for probability in row) for row in ([True, False],))

    with pytest.raises(ValueError, match="boolean"):
        adjust_probabilities_to_source_prior(probabilities, source_labels=[0, 1], classes=[0, 1])
