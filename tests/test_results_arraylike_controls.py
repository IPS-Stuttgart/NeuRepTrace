import numpy as np
import pandas as pd
import pytest

from neureptrace.results import peak_metric_rows, subject_time_metrics


def _result_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": ["s1", "s1"],
            "fold": [0, 1],
            "time": [0.1, 0.1],
            "accuracy": [0.6, 0.8],
            "log_loss": [0.5, 0.4],
            "brier": [0.3, 0.2],
            "ece": [0.1, 0.2],
        }
    )


def test_subject_time_metrics_rejects_series_ece_bins() -> None:
    with pytest.raises(ValueError, match="ece_bins must be a positive integer"):
        subject_time_metrics(_result_frame(), ece_bins=pd.Series([10]))


def test_peak_metric_rows_rejects_series_prefer_time() -> None:
    frame = pd.DataFrame(
        {
            "decoder": ["logistic", "logistic"],
            "time": [-0.1, 0.1],
            "accuracy": [0.8, 0.8],
        }
    )

    with pytest.raises(ValueError, match="prefer_time must be a finite numeric value"):
        peak_metric_rows(frame, "accuracy", ["decoder"], prefer_time=pd.Series([0.0]))


def test_peak_metric_rows_still_accepts_numpy_scalar_prefer_time() -> None:
    frame = pd.DataFrame(
        {
            "decoder": ["logistic", "logistic"],
            "time": [-0.1, 0.1],
            "accuracy": [0.8, 0.8],
        }
    )

    peaks = peak_metric_rows(frame, "accuracy", ["decoder"], prefer_time=np.float64(0.0))

    assert peaks["time"].tolist() == [-0.1]
