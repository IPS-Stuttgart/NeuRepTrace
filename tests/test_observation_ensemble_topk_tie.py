from __future__ import annotations

import numpy as np
import pytest

import neureptrace  # noqa: F401
from neureptrace.observation_ensemble import _top_k_accuracy_from_label_values


def test_observation_ensemble_top_k_ties_use_probability_column_order() -> None:
    probabilities = np.asarray(
        [
            [0.5, 0.5, 0.0],
            [0.2, 0.4, 0.4],
        ],
        dtype=float,
    )
    true_labels = np.asarray([10, 20], dtype=int)
    label_values = (10, 20, 30)

    assert _top_k_accuracy_from_label_values(probabilities, true_labels, label_values, k=1) == pytest.approx(1.0)
