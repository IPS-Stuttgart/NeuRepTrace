from pathlib import Path

import pandas as pd
import pytest

from neureptrace.calibration import aggregate_reliability_bins


def test_aggregate_reliability_bins_prefers_sample_weight_when_available(tmp_path: Path) -> None:
    first = tmp_path / "sub-01_weighted_bins.csv"
    second = tmp_path / "sub-02_weighted_bins.csv"
    pd.DataFrame(
        {
            "decoder": ["logistic"],
            "time": [0.1],
            "bin": [5],
            "bin_left": [0.5],
            "bin_right": [0.6],
            "n_samples": [100],
            "sample_weight": [1.0],
            "accuracy": [1.0],
            "confidence": [0.9],
        }
    ).to_csv(first, index=False)
    pd.DataFrame(
        {
            "decoder": ["logistic"],
            "time": [0.1],
            "bin": [5],
            "bin_left": [0.5],
            "bin_right": [0.6],
            "n_samples": [1],
            "sample_weight": [9.0],
            "accuracy": [0.0],
            "confidence": [0.1],
        }
    ).to_csv(second, index=False)

    aggregated = aggregate_reliability_bins([first, second])

    assert aggregated["n_samples"].tolist() == [101]
    assert aggregated["sample_weight"].tolist() == pytest.approx([10.0])
    assert aggregated["sample_weight_fraction"].tolist() == pytest.approx([1.0])
    assert aggregated["accuracy"].tolist() == pytest.approx([0.1])
    assert aggregated["confidence"].tolist() == pytest.approx([0.18])
    assert aggregated["gap"].tolist() == pytest.approx([-0.08])
