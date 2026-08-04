from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_free_target_prior import apply_target_prior_correction, stabilize_target_class_prior


@pytest.mark.parametrize(
    "prior",
    [
        np.array([[0.60, 0.40]], dtype=float),
        np.array([[0.60], [0.40]], dtype=float),
    ],
)
def test_target_prior_correction_rejects_matrix_shaped_priors(prior: np.ndarray):
    probabilities = np.array([[0.80, 0.20], [0.25, 0.75]], dtype=float)

    with pytest.raises(ValueError, match=r"target prior must have shape \(n_classes,\)"):
        apply_target_prior_correction(probabilities, prior=prior)


@pytest.mark.parametrize(
    "prior",
    [
        np.array([[0.60, 0.40]], dtype=float),
        np.array([[0.60], [0.40]], dtype=float),
    ],
)
def test_target_prior_stabilization_rejects_matrix_shaped_priors(prior: np.ndarray):
    with pytest.raises(ValueError, match=r"target prior must have shape \(n_classes,\)"):
        stabilize_target_class_prior(prior)


def test_target_prior_correction_rejects_complex_probability_arrays():
    probabilities = np.array([[0.80 + 0.05j, 0.20 - 0.05j]], dtype=complex)

    with pytest.raises(ValueError, match="real-valued probability values"):
        apply_target_prior_correction(probabilities)


def test_target_prior_correction_rejects_complex_prior_arrays():
    probabilities = np.array([[0.80, 0.20]], dtype=float)
    prior = np.array([0.60 + 0.05j, 0.40 - 0.05j], dtype=complex)

    with pytest.raises(ValueError, match="target prior.*real-valued probability values"):
        apply_target_prior_correction(probabilities, prior=prior)
