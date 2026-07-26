from __future__ import annotations

import numpy as np
import pytest

from neureptrace.metrics import (
    weighted_brier_score_multiclass,
    weighted_expected_calibration_error,
    weighted_negative_log_likelihood,
    weighted_reliability_bins,
    weighted_top_k_accuracy,
)


@pytest.mark.parametrize(
    "metric",
    [
        weighted_brier_score_multiclass,
        weighted_expected_calibration_error,
        weighted_negative_log_likelihood,
        weighted_reliability_bins,
        weighted_top_k_accuracy,
    ],
    ids=["brier", "ece", "nll", "reliability", "top-k"],
)
def test_weighted_probability_metrics_reject_complex_class_indices(metric) -> None:
    probabilities = np.asarray([[0.8, 0.2], [0.3, 0.7]])
    labels = np.asarray([0.0 + 1.0j, 1.0 + 0.0j], dtype=np.complex128)

    with pytest.raises(ValueError, match="complex"):
        metric(probabilities, labels, np.ones(2))


def test_weighted_probability_metrics_reject_one_pass_complex_class_indices() -> None:
    probabilities = np.asarray([[0.8, 0.2], [0.3, 0.7]])
    labels = (label for label in [0.0 + 1.0j, 1.0 + 0.0j])

    with pytest.raises(ValueError, match="complex"):
        weighted_brier_score_multiclass(probabilities, labels, np.ones(2))
