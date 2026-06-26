from __future__ import annotations

import numpy as np
import pytest

from neureptrace.metrics.weighted import validate_sample_weight, weighted_brier_score_multiclass


@pytest.mark.parametrize(
    "sample_weight",
    [
        np.array([True, False]),
        np.array([1.0, True], dtype=object),
        [np.bool_(True), 1.0],
    ],
)
def test_validate_sample_weight_rejects_boolean_values(sample_weight) -> None:
    with pytest.raises(ValueError, match="not boolean"):
        validate_sample_weight(sample_weight, 2)


def test_weighted_probability_metrics_reject_boolean_sample_weights() -> None:
    probabilities = np.array([[0.7, 0.3], [0.4, 0.6]])
    labels = np.array([0, 1])

    with pytest.raises(ValueError, match="not boolean"):
        weighted_brier_score_multiclass(probabilities, labels, np.array([True, False]))
