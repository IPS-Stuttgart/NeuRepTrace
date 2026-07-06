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


def test_target_prior_accepts_generator_probability_rows():
    probabilities = (row for row in ([0.9, 0.1], [0.7, 0.3]))

    prior = estimate_target_class_prior(probabilities)

    assert np.allclose(prior, [0.8, 0.2])


def test_target_prior_correction_accepts_generator_rows_and_prior():
    probabilities = (row for row in ([0.9, 0.1], [0.7, 0.3]))
    prior = (value for value in [0.6, 0.4])

    corrected, target_prior = apply_target_prior_correction(probabilities, mode="none", prior=prior)

    assert np.allclose(corrected, [[0.9, 0.1], [0.7, 0.3]])
    assert np.allclose(target_prior, [0.6, 0.4])


def test_target_prior_rejects_generator_boolean_probability_values():
    probabilities = (row for row in ([True, False], [False, True]))

    with pytest.raises(ValueError, match="not boolean"):
        estimate_target_class_prior(probabilities)


def test_target_prior_rejects_boolean_supplied_prior_values():
    probabilities = np.array([[0.8, 0.2]], dtype=float)
    prior = (value for value in [True, False])

    with pytest.raises(ValueError, match="not boolean"):
        apply_target_prior_correction(probabilities, mode="none", prior=prior)


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
