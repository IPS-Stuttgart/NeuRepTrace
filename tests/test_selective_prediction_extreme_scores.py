from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.selective_prediction import selective_predict


def test_selective_prediction_normalizes_extreme_finite_scores_without_overflow() -> None:
    maximum = np.finfo(np.float64).max

    result = selective_predict(
        [
            [maximum, maximum],
            [maximum, maximum / 2.0],
        ]
    )

    assert result.probabilities.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])
    assert result.probabilities.tolist() == pytest.approx(
        [
            [0.5, 0.5],
            [2.0 / 3.0, 1.0 / 3.0],
        ]
    )
    assert result.confidence.tolist() == pytest.approx([0.5, 2.0 / 3.0])
    assert result.margin.tolist() == pytest.approx([0.0, 1.0 / 3.0])
