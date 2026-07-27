from __future__ import annotations

import numpy as np
import pandas as pd
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
    "metric",
    [
        validate_probability_inputs,
        brier_score_multiclass,
        expected_calibration_error,
        negative_log_likelihood,
        reliability_bins,
        top_k_accuracy,
    ],
    ids=["validator", "brier", "ece", "nll", "reliability", "top-k"],
)
def test_public_probability_metrics_reject_boolean_dataframe(metric) -> None:
    probabilities = pd.DataFrame(
        {
            "class_0": [True, False],
            "class_1": [False, True],
        }
    )

    with pytest.raises(ValueError, match="probabilities must contain numeric probability values, not boolean flags"):
        metric(probabilities, np.array([0, 1]))
