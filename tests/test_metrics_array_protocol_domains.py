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
def test_public_probability_metrics_reject_complex_dataframe(metric) -> None:
    probabilities = pd.DataFrame(
        {
            "class_0": [0.8 + 0.1j, 0.3 - 0.2j],
            "class_1": [0.2 - 0.1j, 0.7 + 0.2j],
        }
    )

    with pytest.raises(ValueError, match="probabilities must contain real-valued probability values, not complex values"):
        metric(probabilities, np.array([0, 1]))


def test_probability_validator_accepts_real_dataframe() -> None:
    probabilities = pd.DataFrame(
        {
            "class_0": [0.8, 0.3],
            "class_1": [0.2, 0.7],
        }
    )

    validated, labels = validate_probability_inputs(probabilities, np.array([0, 1]))

    np.testing.assert_allclose(validated, probabilities.to_numpy())
    np.testing.assert_array_equal(labels, np.array([0, 1]))
