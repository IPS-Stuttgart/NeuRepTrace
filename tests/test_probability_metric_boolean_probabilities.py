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
    "probabilities",
    [
        np.array([[True, False], [False, True]]),
        np.array([[np.bool_(True), 0.0], [0.0, 1.0]], dtype=object),
        [[True, False], [False, True]],
    ],
)
def test_validate_probability_inputs_rejects_boolean_probabilities(probabilities):
    with pytest.raises(ValueError, match="probabilities must contain numeric probability values, not boolean flags"):
        validate_probability_inputs(probabilities, np.array([0, 1]))


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
def test_probability_metrics_reject_boolean_probabilities(metric):
    with pytest.raises(ValueError, match="probabilities must contain numeric probability values, not boolean flags"):
        metric(np.array([[True, False], [False, True]]), np.array([0, 1]))
