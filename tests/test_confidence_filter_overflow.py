from __future__ import annotations

import numpy as np

from neureptrace.decoding.confidence_filter import confidence_filter, probability_entropy


def test_confidence_filter_normalizes_extreme_finite_rows_without_overflow() -> None:
    maximum = np.finfo(np.float64).max
    rows = np.asarray(
        [
            [maximum, maximum],
            [maximum, maximum / 3.0],
        ],
        dtype=np.float64,
    )

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        result = confidence_filter(rows)
        entropy = probability_entropy(rows)

    assert result.predicted_index.tolist() == [0, 0]
    assert np.allclose(result.confidence, [0.5, 0.75])
    assert np.allclose(result.margin, [0.0, 0.5])
    assert np.all(np.isfinite(entropy))
    assert np.isclose(entropy[0], 1.0)
