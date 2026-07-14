from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.target_confidence_selection import select_target_confident_predictions


def test_target_confidence_selection_rejects_zero_mass_rows_before_epsilon_floor() -> None:
    with pytest.raises(ValueError, match="positive mass"):
        select_target_confident_predictions([[0.0, 0.0]])


def test_target_confidence_selection_normalizes_extreme_finite_rows_without_overflow() -> None:
    scores = np.asarray(
        [
            [1.0e308, 1.0e308],
            [1.0e308, 1.0e307],
        ],
        dtype=float,
    )

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = select_target_confident_predictions(
            scores,
            config={"min_confidence": 0.8},
        )

    expected = np.asarray([[0.5, 0.5], [10.0 / 11.0, 1.0 / 11.0]])
    np.testing.assert_allclose(result.probabilities, expected, rtol=1.0e-6)
    np.testing.assert_allclose(result.confidence, np.max(expected, axis=1), rtol=1.0e-6)
    assert result.predictions.tolist() == [1, 0]
    assert result.keep_mask.tolist() == [False, True]
