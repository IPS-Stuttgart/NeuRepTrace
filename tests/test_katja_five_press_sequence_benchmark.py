from __future__ import annotations

import numpy as np
import pytest

from neureptrace.katja_five_press_sequence_benchmark import (
    fixed_first_press_accuracy,
    scored_variable_press_accuracy,
)


def test_five_press_primary_endpoint_excludes_first_press() -> None:
    labels = np.asarray([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
    positions = np.tile(np.arange(1, 6), (2, 1))
    predictions = labels.copy()
    predictions[:, 0] = np.asarray([3, 2])

    assert scored_variable_press_accuracy(predictions, labels, positions) == 1.0
    assert fixed_first_press_accuracy(predictions, labels, positions) == 0.0


def test_five_press_scoring_rejects_misaligned_arrays() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        scored_variable_press_accuracy(
            np.zeros((1, 5), dtype=int),
            np.zeros((1, 4), dtype=int),
            np.ones((1, 5), dtype=int),
        )


def test_five_press_variable_accuracy_uses_four_events_per_trial() -> None:
    labels = np.asarray([[0, 1, 2, 3, 4]])
    positions = np.asarray([[1, 2, 3, 4, 5]])
    predictions = np.asarray([[0, 1, 2, 0, 0]])

    assert scored_variable_press_accuracy(predictions, labels, positions) == 0.5
    assert fixed_first_press_accuracy(predictions, labels, positions) == 1.0
