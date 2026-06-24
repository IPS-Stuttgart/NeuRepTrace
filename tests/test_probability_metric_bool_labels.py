from __future__ import annotations

import numpy as np
import pytest

from neureptrace.metrics import (
    brier_score_multiclass,
    expected_calibration_error,
    negative_log_likelihood,
    reliability_bins,
    top_k_accuracy,
    validate_probability_inputs,
)


@pytest.mark.parametrize(
    "bad_labels",
    [
        np.array([True, False]),
        np.array([0, True], dtype=object),
        np.array([np.bool_(True), np.bool_(False)], dtype=object),
    ],
)
def test_validate_probability_inputs_rejects_boolean_labels(bad_labels: np.ndarray) -> None:
    probabilities = np.array([[0.6, 0.4], [0.3, 0.7]])

    with pytest.raises(ValueError, match="labels must contain integer class indices"):
        validate_probability_inputs(probabilities, bad_labels)


@pytest.mark.parametrize(
    "metric",
    [
        expected_calibration_error,
        reliability_bins,
        brier_score_multiclass,
        negative_log_likelihood,
        top_k_accuracy,
    ],
)
@pytest.mark.parametrize(
    "bad_labels",
    [
        np.array([True, False]),
        np.array([0, True], dtype=object),
        np.array([np.bool_(True), np.bool_(False)], dtype=object),
    ],
)
def test_probability_metrics_reject_boolean_label_indices(metric, bad_labels: np.ndarray) -> None:
    probabilities = np.array([[0.6, 0.4], [0.3, 0.7]])

    with pytest.raises(ValueError, match="labels must contain integer class indices"):
        metric(probabilities, bad_labels)
