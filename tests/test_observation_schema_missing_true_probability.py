from __future__ import annotations

import numpy as np
import pandas as pd

from neureptrace.observation_schema import validate_probability_observations


def test_canonical_profile_flags_missing_true_probability_value() -> None:
    frame = pd.DataFrame(
        {
            "time": [0.0],
            "decoder": ["logistic"],
            "backend": ["sklearn"],
            "emission_mode": ["calibrated"],
            "split_id": ["split-001"],
            "seed": [0],
            "train_time": [0.0],
            "test_time": [0.0],
            "preprocessing_hash": ["pre-123"],
            "model_hash": ["model-456"],
            "true_label": [1],
            "probability_true_class": [0.25],
            "prob_class_0": [1.0],
            "prob_class_1": [np.nan],
        }
    )

    report = validate_probability_observations(frame, profile="canonical")

    assert not report.is_valid
    assert any(
        issue.code == "missing_true_label_probability_value" and issue.column == "prob_class_1"
        for issue in report.errors
    )
