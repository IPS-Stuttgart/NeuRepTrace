from __future__ import annotations

import numpy as np
import pytest

from neureptrace.metrics.weighted import (
    validate_sample_weight,
    weighted_brier_score_multiclass,
    weighted_negative_log_likelihood,
)


def test_weighted_metrics_rescale_positive_subnormal_weights() -> None:
    probabilities = np.array(
        [
            [0.99, 0.01],
            [0.01, 0.99],
        ]
    )
    labels = np.array([0, 0])
    tiny = np.nextafter(0.0, 1.0)
    sample_weight = np.array([tiny, tiny])

    np.testing.assert_allclose(validate_sample_weight(sample_weight, 2), [1.0, 1.0])

    expected_brier = np.mean([2.0 * 0.01**2, 2.0 * 0.99**2])
    expected_nll = -np.mean(np.log([0.99, 0.01]))
    assert weighted_brier_score_multiclass(probabilities, labels, sample_weight) == pytest.approx(expected_brier)
    assert weighted_negative_log_likelihood(probabilities, labels, sample_weight) == pytest.approx(expected_nll)
