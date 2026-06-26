from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neureptrace.calibration import aggregate_reliability_bins, summarize_calibration_metrics


def test_aggregate_reliability_bins_accepts_empty_bins_without_metrics(tmp_path: Path):
    path = tmp_path / "empty_calibration_bins.csv"
    pd.DataFrame(
        {
            "time": [0.1],
            "bin": [0],
            "bin_left": [0.0],
            "bin_right": [0.1],
            "n_samples": [0],
            "accuracy": [np.nan],
            "confidence": [np.nan],
        }
    ).to_csv(path, index=False)

    aggregated = aggregate_reliability_bins([path])

    assert aggregated["n_samples"].tolist() == [0]
    assert aggregated["accuracy"].isna().tolist() == [True]
    assert aggregated["confidence"].isna().tolist() == [True]
    assert aggregated["gap"].isna().tolist() == [True]


def test_aggregate_reliability_bins_rejects_non_empty_bins_without_metrics(tmp_path: Path):
    path = tmp_path / "missing_non_empty_calibration_bins.csv"
    pd.DataFrame(
        {
            "time": [0.1],
            "bin": [0],
            "bin_left": [0.0],
            "bin_right": [0.1],
            "n_samples": [1],
            "accuracy": [np.nan],
            "confidence": [0.6],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing values.*accuracy.*non-empty"):
        aggregate_reliability_bins([path])


def test_summarize_calibration_metrics_rejects_boolean_numeric_values():
    summary = pd.DataFrame(
        {
            "time": [0.0, 0.2],
            "accuracy_mean": [0.5, 0.6],
            "log_loss_mean": [0.7, 0.6],
            "brier_mean": [0.25, 0.2],
            "ece_mean": [0.12, 0.08],
            "n_subjects": [True, True],
        }
    )

    with pytest.raises(ValueError, match="boolean values.*n_subjects"):
        summarize_calibration_metrics(summary)


def test_aggregate_reliability_bins_rejects_boolean_numeric_values(tmp_path: Path):
    path = tmp_path / "boolean_calibration_bins.csv"
    pd.DataFrame(
        {
            "time": [0.1],
            "bin": [True],
            "bin_left": [0.0],
            "bin_right": [0.1],
            "n_samples": [1],
            "accuracy": [1.0],
            "confidence": [0.6],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="boolean values.*bin"):
        aggregate_reliability_bins([path])


def test_aggregate_reliability_bins_rejects_boolean_sample_weight(tmp_path: Path):
    path = tmp_path / "boolean_sample_weight_bins.csv"
    pd.DataFrame(
        {
            "time": [0.1],
            "bin": [0],
            "bin_left": [0.0],
            "bin_right": [0.1],
            "n_samples": [1],
            "accuracy": [1.0],
            "confidence": [0.6],
            "sample_weight": [True],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="boolean values.*sample_weight"):
        aggregate_reliability_bins([path])
