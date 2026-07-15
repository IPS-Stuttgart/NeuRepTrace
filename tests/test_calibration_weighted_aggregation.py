from pathlib import Path

import numpy as np
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


def test_aggregate_reliability_bins_normalizes_sample_weight_fraction_per_time(tmp_path: Path) -> None:
    bins_csv = tmp_path / "weighted_bins.csv"
    pd.DataFrame(
        {
            "decoder": ["logistic", "logistic", "logistic", "logistic"],
            "time": [0.1, 0.1, 0.2, 0.2],
            "bin": [0, 1, 0, 1],
            "bin_left": [0.0, 0.5, 0.0, 0.5],
            "bin_right": [0.5, 1.0, 0.5, 1.0],
            "n_samples": [2, 8, 1, 3],
            "sample_weight": [2.0, 8.0, 1.0, 3.0],
            "accuracy": [1.0, 0.5, 0.0, 0.75],
            "confidence": [0.25, 0.75, 0.2, 0.8],
        }
    ).to_csv(bins_csv, index=False)

    aggregated = aggregate_reliability_bins([bins_csv]).sort_values(["time", "bin"]).reset_index(drop=True)

    assert aggregated["sample_weight_fraction"].tolist() == pytest.approx([0.2, 0.8, 0.25, 0.75])
    assert aggregated.groupby("time")["sample_weight_fraction"].sum().tolist() == pytest.approx([1.0, 1.0])


def test_aggregate_reliability_bins_preserves_large_finite_weight_fractions(tmp_path: Path) -> None:
    bins_csv = tmp_path / "large_weighted_bins.csv"
    pd.DataFrame(
        {
            "decoder": ["logistic", "logistic"],
            "time": [0.1, 0.1],
            "bin": [0, 1],
            "bin_left": [0.0, 0.5],
            "bin_right": [0.5, 1.0],
            "n_samples": [1, 1],
            "sample_weight": [1e308, 1e308],
            "accuracy": [1.0, 0.0],
            "confidence": [0.75, 0.25],
        }
    ).to_csv(bins_csv, index=False)

    with np.errstate(over="raise", invalid="raise"):
        aggregated = aggregate_reliability_bins([bins_csv]).sort_values("bin").reset_index(drop=True)

    assert aggregated["sample_weight"].tolist() == pytest.approx([1e308, 1e308])
    assert aggregated["sample_weight_fraction"].tolist() == pytest.approx([0.5, 0.5])
    assert aggregated["sample_weight_fraction"].sum() == pytest.approx(1.0)


def test_aggregate_reliability_bins_rejects_positive_weight_for_empty_bins(tmp_path: Path) -> None:
    bins_csv = tmp_path / "bad_weighted_empty_bins.csv"
    pd.DataFrame(
        {
            "decoder": ["logistic"],
            "time": [0.1],
            "bin": [0],
            "bin_left": [0.0],
            "bin_right": [0.5],
            "n_samples": [0],
            "sample_weight": [1.0],
            "accuracy": [float("nan")],
            "confidence": [float("nan")],
        }
    ).to_csv(bins_csv, index=False)

    with pytest.raises(ValueError, match="positive sample_weight.*empty reliability bin"):
        aggregate_reliability_bins([bins_csv])
