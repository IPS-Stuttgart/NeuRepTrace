from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_free_target_prior import apply_target_prior_correction, stabilize_target_class_prior


_PROBABILITIES = np.array([[0.8, 0.2], [0.7, 0.3]], dtype=float)


def test_target_prior_strength_rejects_size_one_array():
    with pytest.raises(ValueError, match="target_prior_strength"):
        apply_target_prior_correction(_PROBABILITIES, strength=np.array([0.5]))


def test_target_prior_smoothing_rejects_boolean_array():
    with pytest.raises(ValueError, match="target_prior_smoothing"):
        apply_target_prior_correction(_PROBABILITIES, smoothing=np.array(True))


def test_target_prior_floor_rejects_size_one_array():
    with pytest.raises(ValueError, match="target_prior_floor"):
        stabilize_target_class_prior(np.array([0.7, 0.3], dtype=float), floor=np.array([0.1]))


def test_target_prior_scalar_controls_accept_numpy_numeric_scalars():
    corrected, prior = apply_target_prior_correction(
        _PROBABILITIES,
        strength=np.float64(0.5),
        smoothing=np.array(0.25),
        floor=np.array(0.01),
    )

    assert corrected.shape == _PROBABILITIES.shape
    assert np.allclose(corrected.sum(axis=1), 1.0)
    assert np.allclose(prior.sum(), 1.0)
