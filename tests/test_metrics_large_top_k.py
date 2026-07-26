from __future__ import annotations

import numpy as np

from neureptrace.metrics import top_k_accuracy, weighted_top_k_accuracy


def test_top_k_metrics_preserve_large_exact_integer_limits() -> None:
    probabilities = np.asarray(
        [
            [0.9, 0.1],
            [0.2, 0.8],
        ]
    )
    labels = np.asarray([0, 1])
    huge_k = 10**400

    assert top_k_accuracy(probabilities, labels, k=huge_k) == 1.0
    assert weighted_top_k_accuracy(probabilities, labels, [1.0, 3.0], k=huge_k) == 1.0
