from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neureptrace.temporal_model import fit_sticky_switching_model, fit_temporal_models, read_probability_observations
from neureptrace.temporal_smoothing import metrics_from_probability_observations


def _valid_probability_observations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": ["sub-01", "sub-01"],
            "fold": [0, 0],
            "decoder": ["logistic", "logistic"],
            "emission_mode": ["calibrated", "calibrated"],
            "time": [0.1, 0.2],
            "sequence_id": [0, 0],
            "true_label": [0, 1],
            "prob_class_0": [0.8, 0.2],
            "prob_class_1": [0.2, 0.8],
        }
    )


def _write_valid_probability_observations(tmp_path: Path) -> Path:
    csv_path = tmp_path / "valid_probability_observations.csv"
    _valid_probability_observations().to_csv(csv_path, index=False)
    return csv_path


def test_temporal_model_rejects_boolean_time_values(tmp_path: Path) -> None:
    frame = _valid_probability_observations()
    frame["time"] = [True, False]
    csv_path = tmp_path / "bool_time_observations.csv"
    frame.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="time values must be numeric, not boolean"):
        read_probability_observations([csv_path])


def test_temporal_model_rejects_boolean_probability_values(tmp_path: Path) -> None:
    frame = _valid_probability_observations()
    frame["prob_class_0"] = [True, False]
    frame["prob_class_1"] = [False, True]
    csv_path = tmp_path / "bool_probability_observations.csv"
    frame.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="numeric probabilities, not boolean"):
        read_probability_observations([csv_path])


def test_temporal_model_rejects_boolean_array_time_window_bounds(tmp_path: Path) -> None:
    csv_path = _write_valid_probability_observations(tmp_path)

    with pytest.raises(ValueError, match="effect_window must be finite"):
        fit_temporal_models(
            [csv_path],
            effect_window=(0.1, np.asarray(True)),
            n_permutations=0,
            stay_grid_size=10,
        )


def test_temporal_model_rejects_boolean_array_integer_options(tmp_path: Path) -> None:
    csv_path = _write_valid_probability_observations(tmp_path)

    with pytest.raises(ValueError, match="n_permutations must be a non-negative integer"):
        fit_temporal_models(
            [csv_path],
            n_permutations=np.asarray(True),
            stay_grid_size=10,
        )


def test_temporal_model_rejects_array_valued_stay_grid_size() -> None:
    sequences = [np.asarray([[0.8, 0.2], [0.7, 0.3]], dtype=float)]

    with pytest.raises(ValueError, match="stay_grid_size must be an integer"):
        fit_sticky_switching_model(sequences, stay_grid_size=np.asarray([10]))


def test_temporal_smoothing_metrics_reject_boolean_true_labels() -> None:
    observations = _valid_probability_observations()
    observations["true_label"] = [True, False]

    with pytest.raises(ValueError, match="true_label values must be numeric integer labels, not boolean"):
        metrics_from_probability_observations(observations)


def test_temporal_smoothing_metrics_reject_boolean_probability_values() -> None:
    observations = _valid_probability_observations()
    observations["prob_class_0"] = [True, False]
    observations["prob_class_1"] = [False, True]

    with pytest.raises(ValueError, match="prob_class_0 values must be numeric probabilities, not boolean"):
        metrics_from_probability_observations(observations)
