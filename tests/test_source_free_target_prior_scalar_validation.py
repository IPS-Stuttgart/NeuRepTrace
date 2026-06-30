from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_free_target_prior import apply_target_prior_correction, stabilize_target_class_prior


@pytest.mark.parametrize("strength", [np.asarray(0.5), np.array([0.5])])
def test_target_prior_strength_rejects_array_scalar_controls(strength):
    with pytest.raises(ValueError, match="target_prior_strength"):
        apply_target_prior_correction(np.array([[0.8, 0.2]], dtype=float), strength=strength)


@pytest.mark.parametrize("smoothing", [np.asarray(0.5), np.array([0.5])])
def test_target_prior_smoothing_rejects_array_scalar_controls(smoothing):
    with pytest.raises(ValueError, match="target_prior_smoothing"):
        apply_target_prior_correction(np.array([[0.8, 0.2]], dtype=float), smoothing=smoothing)


@pytest.mark.parametrize("floor", [np.asarray(0.05), np.array([0.05])])
def test_target_prior_floor_rejects_array_scalar_controls(floor):
    with pytest.raises(ValueError, match="target_prior_floor"):
        apply_target_prior_correction(np.array([[0.8, 0.2]], dtype=float), floor=floor)


def test_target_prior_controls_accept_numpy_numeric_scalars():
    corrected, prior = apply_target_prior_correction(
        np.array([[0.8, 0.2]], dtype=float),
        strength=np.float64(0.5),
        smoothing=np.float64(0.1),
        floor=np.float64(0.05),
    )

    assert corrected.shape == (1, 2)
    assert np.allclose(corrected.sum(axis=1), 1.0)
    assert np.allclose(prior.sum(), 1.0)


def test_stabilize_target_prior_rejects_array_scalar_controls():
    with pytest.raises(ValueError, match="target_prior_smoothing"):
        stabilize_target_class_prior(np.array([0.8, 0.2]), smoothing=np.array([0.1]))
    with pytest.raises(ValueError, match="target_prior_floor"):
        stabilize_target_class_prior(np.array([0.8, 0.2]), floor=np.array([0.05]))
