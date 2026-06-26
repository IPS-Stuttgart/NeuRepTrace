from __future__ import annotations

import numpy as np
import pytest

from neureptrace.metrics import negative_log_likelihood


@pytest.mark.parametrize("eps", [0.0, -1.0, np.inf, np.nan, False, True])
def test_negative_log_likelihood_rejects_malformed_eps(eps: object) -> None:
    probabilities = np.array([[0.8, 0.2], [0.1, 0.9]])
    labels = np.array([0, 1])

    with pytest.raises(ValueError, match="eps must be a positive finite value"):
        negative_log_likelihood(probabilities, labels, eps=eps)
