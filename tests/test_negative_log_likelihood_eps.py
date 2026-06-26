from __future__ import annotations

import numpy as np
import pytest

from neureptrace.metrics import negative_log_likelihood
from neureptrace.metrics.weighted import weighted_negative_log_likelihood


@pytest.mark.parametrize("bad_eps", [0.0, -1.0, np.inf, np.nan, True, np.bool_(False)])
def test_negative_log_likelihood_rejects_malformed_eps(bad_eps: object) -> None:
    probabilities = np.array([[0.7, 0.3]])
    labels = np.array([0])

    with pytest.raises(ValueError, match="eps must be a positive finite value"):
        negative_log_likelihood(probabilities, labels, eps=bad_eps)


@pytest.mark.parametrize("bad_eps", [0.0, -1.0, np.inf, np.nan, True, np.bool_(False)])
def test_weighted_negative_log_likelihood_rejects_malformed_eps(bad_eps: object) -> None:
    probabilities = np.array([[0.7, 0.3]])
    labels = np.array([0])
    sample_weight = np.array([1.0])

    with pytest.raises(ValueError, match="eps must be a positive finite value"):
        weighted_negative_log_likelihood(probabilities, labels, sample_weight, eps=bad_eps)
