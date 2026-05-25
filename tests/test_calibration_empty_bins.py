from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neureptrace.calibration import aggregate_reliability_bins


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
