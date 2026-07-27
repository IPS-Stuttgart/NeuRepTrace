import numpy as np
import pandas as pd
import pytest

from neureptrace.temporal_smoothing import metrics_from_probability_observations


def _observations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [0.1, 0.1],
            "true_label": [0, 1],
            "prob_class_0": [0.8, 0.2],
            "prob_class_1": [0.2, 0.8],
        }
    )


def test_temporal_smoothing_metrics_reject_complex_true_labels() -> None:
    observations = _observations()
    observations["true_label"] = np.asarray([0.0 + 1.0j, 1.0 + 0.0j])

    with pytest.raises(ValueError, match="true_label values must be real-valued integer labels"):
        metrics_from_probability_observations(observations)


def test_temporal_smoothing_metrics_reject_complex_probabilities() -> None:
    observations = _observations()
    observations["prob_class_0"] = np.asarray([0.8 + 0.1j, 0.2 + 0.0j])

    with pytest.raises(ValueError, match="prob_class_0 values must be real-valued probabilities"):
        metrics_from_probability_observations(observations)
