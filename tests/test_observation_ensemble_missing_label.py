from __future__ import annotations

import pandas as pd
import pytest

from neureptrace.observation_ensemble import ensemble_probability_observations


def test_ensemble_probability_observations_rejects_true_labels_missing_from_probability_columns() -> None:
    observations = pd.DataFrame(
        [
            {
                "decoder": decoder,
                "emission_mode": "calibrated",
                "time": 0.1,
                "sample_index": 0,
                "sequence_id": 0,
                "true_label": 2,
                "true_class": "two",
                "class_0": "zero",
                "class_1": "one",
                "prob_class_0": probability_0,
                "prob_class_1": 1.0 - probability_0,
            }
            for decoder, probability_0 in (("logistic", 0.55), ("linear_svm", 0.45))
        ]
    )

    with pytest.raises(ValueError, match="true_label values must index probability labels"):
        ensemble_probability_observations(observations, baseline_window=None)
