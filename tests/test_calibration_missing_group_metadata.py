from pathlib import Path

import numpy as np
import pandas as pd

from neureptrace.calibration import aggregate_reliability_bins


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
