from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_free_target_prior import estimate_target_class_prior


def test_estimated_target_prior_sums_to_one_when_a_class_is_absent():
    prior = estimate_target_class_prior(np.array([[1.0, 0.0], [1.0, 0.0]], dtype=float))

    assert np.allclose(prior.sum(), 1.0)
    assert np.all(prior > 0.0)
    assert prior[1] < 1e-9
