from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.target_confidence_gate import (
    gate_target_probabilities_by_confidence,
    target_confidence_scores,
)


def test_target_confidence_gate_rejects_zero_mass_rows_before_epsilon_floor() -> None:
    with pytest.raises(ValueError, match="positive mass"):
        target_confidence_scores([[0.0, 0.0]])

    with pytest.raises(ValueError, match="positive mass"):
        gate_target_probabilities_by_confidence([[0.0, 0.0]])


def test_target_confidence_gate_normalizes_extreme_finite_rows_without_overflow() -> None:
    scores = np.asarray(
        [
            [1.0e308, 1.0e308],
            [1.0e308, 1.0e307],
        ],
        dtype=float,
    )

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        confidence = target_confidence_scores(scores)
        result = gate_target_probabilities_by_confidence(
            scores,
            config={"confidence_threshold": 0.8},
        )

    expected = np.asarray([[0.5, 0.5], [10.0 / 11.0, 1.0 / 11.0]])
    np.testing.assert_allclose(result.probabilities, expected, rtol=1.0e-6)
    np.testing.assert_allclose(confidence, np.max(expected, axis=1))
    assert result.predictions.tolist() == [0, 0]
    assert result.accepted_mask.tolist() == [False, True]
