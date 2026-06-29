from pathlib import Path

import pandas as pd
import pytest

from neureptrace.response_window_ensemble import run_response_window_ensemble


def _minimal_response_window_observations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject": "sub-01",
                "fold": "sub-01",
                "decoder": "base",
                "emission_mode": "calibrated",
                "time": 0.088,
                "test_time": 0.088,
                "sample_index": 0,
                "sequence_id": 0,
                "true_label": 0,
                "true_class": "class-0",
                "predicted_label": 0,
                "predicted_class": "class-0",
                "probability_true_class": 0.8,
                "confidence": 0.8,
                "class_0": "class-0",
                "class_1": "class-1",
                "prob_class_0": 0.8,
                "prob_class_1": 0.2,
            },
            {
                "subject": "sub-01",
                "fold": "sub-01",
                "decoder": "base",
                "emission_mode": "calibrated",
                "time": 0.088,
                "test_time": 0.088,
                "sample_index": 1,
                "sequence_id": 1,
                "true_label": 1,
                "true_class": "class-1",
                "predicted_label": 1,
                "predicted_class": "class-1",
                "probability_true_class": 0.7,
                "confidence": 0.7,
                "class_0": "class-0",
                "class_1": "class-1",
                "prob_class_0": 0.3,
                "prob_class_1": 0.7,
            },
        ]
    )


def test_response_window_rejects_boolean_true_labels(tmp_path: Path):
    observations = _minimal_response_window_observations()
    observations["true_label"] = [False, True]
    csv_path = tmp_path / "observations.csv"
    observations.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="true_label.*booleans"):
        run_response_window_ensemble([csv_path], mode="uniform", response_times=(0.088,))


def test_response_window_rejects_boolean_probability_values(tmp_path: Path):
    observations = _minimal_response_window_observations()
    observations["prob_class_0"] = [True, False]
    observations["prob_class_1"] = [False, True]
    csv_path = tmp_path / "observations.csv"
    observations.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Probability observations.*booleans"):
        run_response_window_ensemble([csv_path], mode="uniform", response_times=(0.088,))
