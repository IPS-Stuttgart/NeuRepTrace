from pathlib import Path

import pandas as pd
import pytest

from neureptrace.calibration import aggregate_reliability_bins, summarize_calibration_metrics


def _summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decoder": ["logistic", "logistic"],
            "time": [-0.05, 0.15],
            "accuracy_mean": [0.50, 0.60],
            "log_loss_mean": [0.70, 0.66],
            "brier_mean": [0.50, 0.47],
            "ece_mean": [0.09, 0.06],
            "n_subjects": [5, 5],
        }
    )


@pytest.mark.parametrize("column", ["time", "accuracy_mean", "log_loss_mean", "brier_mean", "ece_mean", "n_subjects"])
def test_summarize_calibration_metrics_rejects_boolean_numeric_columns(column: str):
    frame = _summary_frame()
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = True

    with pytest.raises(ValueError, match=f"boolean values in numeric column '{column}'"):
        summarize_calibration_metrics(frame)


@pytest.mark.parametrize("column", ["time", "bin", "bin_left", "bin_right", "n_samples", "accuracy", "confidence"])
def test_aggregate_reliability_bins_rejects_boolean_numeric_columns(tmp_path: Path, column: str):
    path = tmp_path / "bad_bool_calibration_bins.csv"
    frame = pd.DataFrame(
        {
            "time": [0.1],
            "bin": [5],
            "bin_left": [0.5],
            "bin_right": [0.6],
            "n_samples": [10],
            "accuracy": [0.8],
            "confidence": [0.6],
        }
    )
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = True
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match=f"boolean values in numeric column '{column}'"):
        aggregate_reliability_bins([path])


def test_aggregate_reliability_bins_rejects_boolean_sample_weight(tmp_path: Path):
    path = tmp_path / "bad_bool_sample_weight_calibration_bins.csv"
    pd.DataFrame(
        {
            "time": [0.1],
            "bin": [5],
            "bin_left": [0.5],
            "bin_right": [0.6],
            "n_samples": [10],
            "accuracy": [0.8],
            "confidence": [0.6],
            "sample_weight": [True],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="boolean values in numeric column 'sample_weight'"):
        aggregate_reliability_bins([path])
