from __future__ import annotations

import numpy as np
import pytest

from neureptrace.metrics.weighted import weighted_expected_calibration_error, weighted_reliability_bins


def test_weighted_metric_bin_count_rejects_fractional_values() -> None:
    probabilities = np.array([[0.7, 0.3], [0.4, 0.6]])
    labels = np.array([0, 1])
    sample_weight = np.array([1.0, 1.0])

    with pytest.raises(ValueError, match="n_bins must be a positive integer"):
        weighted_expected_calibration_error(probabilities, labels, sample_weight, n_bins=2.5)

    with pytest.raises(ValueError, match="n_bins must be a positive integer"):
        weighted_reliability_bins(probabilities, labels, sample_weight, n_bins=float("nan"))
