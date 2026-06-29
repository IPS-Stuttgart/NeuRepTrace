from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_free_target_prior import (
    apply_target_prior_correction,
    estimate_target_class_prior,
)


@pytest.mark.parametrize(
    "probabilities",
    [
        np.array([[True, False], [False, True]], dtype=bool),
        np.array([[True, 0.0], [False, 1.0]], dtype=object),
    ],
)
def test_target_prior_rejects_boolean_probability_values(probabilities):
    with pytest.raises(ValueError, match="not boolean"):
        estimate_target_class_prior(probabilities)


def test_target_prior_rejects_materially_negative_probability_values():
    probabilities = np.array([[0.8, 0.2], [0.4, -0.1]], dtype=float)

    with pytest.raises(ValueError, match="non-negative"):
        apply_target_prior_correction(probabilities)


def test_target_prior_allows_tiny_negative_probability_roundoff():
    probabilities = np.array([[1.0, -1e-12], [0.25, 0.75]], dtype=float)

    corrected, prior = apply_target_prior_correction(probabilities, mode="none")

    assert np.all(corrected >= 0.0)
    assert np.allclose(corrected.sum(axis=1), 1.0)
    assert np.allclose(prior.sum(), 1.0)
