from __future__ import annotations

import numpy as np

from neureptrace.decoding.confidence_selection import select_confident_probability_rows


def test_confidence_selection_normalizes_extreme_finite_rows_without_overflow() -> None:
    limit = np.finfo(np.float64).max
    probabilities = np.asarray(
        [
            [limit, limit],
            [limit, limit / 3.0],
        ],
        dtype=float,
    )

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = select_confident_probability_rows(
            probabilities,
            config={"threshold": 0.7},
        )

    np.testing.assert_allclose(result.confidences, [0.5, 0.75])
    np.testing.assert_allclose(result.margins, [0.0, 0.5])
    assert result.predicted_indices.tolist() == [0, 0]
    assert result.selected_mask.tolist() == [False, True]
    assert result.selected_indices.tolist() == [1]
