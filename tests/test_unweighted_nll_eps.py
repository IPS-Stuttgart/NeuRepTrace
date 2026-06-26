from __future__ import annotations

import numpy as np
import pytest

from neureptrace.metrics import negative_log_likelihood


@pytest.mark.parametrize("bad_eps", [0.0, -1e-3, 1.0, 1.5, np.inf, np.nan, True])
def test_negative_log_likelihood_rejects_degenerate_eps(bad_eps: object) -> None:
    probabilities = np.array([[0.7, 0.3], [0.4, 0.6]])
    labels = np.array([0, 1])

    with pytest.raises(ValueError, match="eps"):
        negative_log_likelihood(probabilities, labels, eps=bad_eps)
