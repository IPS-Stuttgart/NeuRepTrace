from pathlib import Path

import numpy as np
import pandas as pd

from neureptrace.calibration import aggregate_reliability_bins, summarize_calibration_metrics


def test_aggregate_reliability_bins_preserves_missing_group_metadata_rows(tmp_path: Path):
    path = tmp_path / "calibration_bins.csv"
    pd.DataFrame(
        {
            "decoder": ["logistic", np.nan],
            "emission_mode": ["calibrated", np.nan],
            "time": [0.1, 0.1],
            "bin": [5, 5],
            "bin_left": [0.5, 0.5],
            "bin_right": [0.6, 0.6],
            "n_samples": [10, 20],
            "accuracy": [0.8, 0.4],
            "confidence": [0.6, 0.5],
        }
    ).to_csv(path, index=False)

    aggregated = aggregate_reliability_bins([path])

    assert aggregated["n_samples"].sum() == 30
    assert set(zip(aggregated["decoder"], aggregated["emission_mode"], strict=True)) == {
        ("logistic", "calibrated"),
        ("overall", "calibrated"),
    }


def test_summarize_calibration_metrics_preserves_missing_group_metadata_rows():
    frame = pd.DataFrame(
        {
            "decoder": ["logistic", "logistic", np.nan, np.nan],
            "emission_mode": ["uncalibrated", "uncalibrated", np.nan, np.nan],
            "time": [-0.05, 0.15, -0.05, 0.15],
            "accuracy_mean": [0.50, 0.60, 0.55, 0.65],
            "log_loss_mean": [0.70, 0.66, 0.68, 0.62],
            "brier_mean": [0.50, 0.47, 0.48, 0.44],
            "ece_mean": [0.09, 0.06, 0.08, 0.05],
            "n_subjects": [5, 5, 3, 3],
        }
    )

    summary = summarize_calibration_metrics(
        frame,
        baseline_window=(-0.1, 0.0),
        effect_window=(0.1, 0.2),
    )

    assert len(summary) == 2
    assert set(zip(summary["decoder"], summary["emission_mode"], strict=True)) == {
        ("logistic", "uncalibrated"),
        ("overall", "calibrated"),
    }
    assert summary["n_subjects"].sum() == 8
