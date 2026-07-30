from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neureptrace.plot_calibration import plot_reliability_diagram, summarize_reliability_curve


def test_summarize_reliability_curve_weights_by_samples():
    bins = pd.DataFrame(
        {
            "decoder": ["logistic", "logistic", "logistic"],
            "time": [0.1, 0.2, 0.2],
            "bin": [1, 1, 2],
            "bin_left": [0.0, 0.0, 0.5],
            "bin_right": [0.5, 0.5, 1.0],
            "n_samples": [1, 3, 2],
            "accuracy": [1.0, 0.0, 0.5],
            "confidence": [0.2, 0.4, 0.8],
        }
    )

    curve = summarize_reliability_curve(bins, time_window=(0.1, 0.2))

    first_bin = curve[curve["bin"] == 1].iloc[0]
    assert first_bin["n_samples"] == 4
    assert first_bin["accuracy"] == 0.25
    assert first_bin["confidence"] == pytest.approx(0.35)


def test_summarize_reliability_curve_prefers_sample_weight():
    bins = pd.DataFrame(
        {
            "decoder": ["logistic", "logistic"],
            "time": [0.1, 0.2],
            "bin": [1, 1],
            "bin_left": [0.0, 0.0],
            "bin_right": [0.5, 0.5],
            "n_samples": [100, 1],
            "sample_weight": [1.0, 9.0],
            "accuracy": [1.0, 0.0],
            "confidence": [0.9, 0.1],
        }
    )

    curve = summarize_reliability_curve(bins)

    assert curve.loc[0, "n_samples"] == 101
    assert curve.loc[0, "accuracy"] == pytest.approx(0.1)
    assert curve.loc[0, "confidence"] == pytest.approx(0.18)
    assert curve.loc[0, "gap"] == pytest.approx(-0.08)


def test_summarize_reliability_curve_falls_back_for_missing_sample_weight():
    bins = pd.DataFrame(
        {
            "decoder": ["logistic", "logistic"],
            "time": [0.1, 0.2],
            "bin": [1, 1],
            "bin_left": [0.0, 0.0],
            "bin_right": [0.5, 0.5],
            "n_samples": [2, 8],
            "sample_weight": [pd.NA, 2.0],
            "accuracy": [1.0, 0.0],
            "confidence": [0.8, 0.2],
        }
    )

    curve = summarize_reliability_curve(bins)

    assert curve.loc[0, "n_samples"] == 10
    assert curve.loc[0, "accuracy"] == pytest.approx(0.5)
    assert curve.loc[0, "confidence"] == pytest.approx(0.5)
    assert curve.loc[0, "gap"] == pytest.approx(0.0)


def test_summarize_reliability_curve_handles_large_sample_weights():
    bins = pd.DataFrame(
        {
            "decoder": ["logistic", "logistic"],
            "time": [0.1, 0.2],
            "bin": [1, 1],
            "bin_left": [0.0, 0.0],
            "bin_right": [0.5, 0.5],
            "n_samples": [1, 1],
            "sample_weight": [1e308, 1e308],
            "accuracy": [1.0, 0.0],
            "confidence": [0.75, 0.25],
        }
    )

    curve = summarize_reliability_curve(bins)

    assert curve.loc[0, "accuracy"] == pytest.approx(0.5)
    assert curve.loc[0, "confidence"] == pytest.approx(0.5)


@pytest.mark.parametrize("sample_weight", [[-1.0, 2.0], [np.inf, 2.0]])
def test_summarize_reliability_curve_rejects_invalid_sample_weight(sample_weight):
    bins = pd.DataFrame(
        {
            "decoder": ["logistic", "logistic"],
            "time": [0.1, 0.2],
            "bin": [1, 1],
            "bin_left": [0.0, 0.0],
            "bin_right": [0.5, 0.5],
            "n_samples": [1, 1],
            "sample_weight": sample_weight,
            "accuracy": [1.0, 0.0],
            "confidence": [0.9, 0.1],
        }
    )

    with pytest.raises(ValueError, match="sample_weight values must be finite and non-negative"):
        summarize_reliability_curve(bins)


@pytest.mark.parametrize("n_samples", [[1, -1], [1.5, 2]])
def test_summarize_reliability_curve_rejects_invalid_sample_counts(n_samples):
    bins = pd.DataFrame(
        {
            "decoder": ["logistic", "logistic"],
            "time": [0.1, 0.2],
            "bin": [1, 1],
            "bin_left": [0.0, 0.0],
            "bin_right": [0.5, 0.5],
            "n_samples": n_samples,
            "accuracy": [1.0, 0.0],
            "confidence": [0.9, 0.1],
        }
    )

    with pytest.raises(ValueError, match="n_samples values must be finite non-negative integers"):
        summarize_reliability_curve(bins)


def test_summarize_reliability_curve_preserves_rows_with_missing_group_labels():
    bins = pd.DataFrame(
        {
            "decoder": [pd.NA],
            "emission_mode": [pd.NA],
            "time": [0.1],
            "bin": [1],
            "bin_left": [0.0],
            "bin_right": [0.5],
            "n_samples": [4],
            "accuracy": [0.75],
            "confidence": [0.6],
        }
    )

    curve = summarize_reliability_curve(bins)

    assert curve[["decoder", "emission_mode"]].to_dict("records") == [
        {"decoder": "overall", "emission_mode": "calibrated"}
    ]
    assert curve.loc[0, "n_samples"] == 4
    assert curve.loc[0, "accuracy"] == 0.75
    assert curve.loc[0, "confidence"] == 0.6


def test_plot_reliability_diagram_writes_png(tmp_path: Path):
    bins_csv = tmp_path / "reliability_bins.csv"
    out_path = tmp_path / "reliability.png"
    pd.DataFrame(
        {
            "decoder": ["logistic", "logistic", "lda", "lda"],
            "time": [0.1, 0.1, 0.1, 0.1],
            "bin": [1, 2, 1, 2],
            "bin_left": [0.0, 0.5, 0.0, 0.5],
            "bin_right": [0.5, 1.0, 0.5, 1.0],
            "n_samples": [10, 20, 8, 16],
            "accuracy": [0.4, 0.8, 0.3, 0.7],
            "confidence": [0.35, 0.75, 0.25, 0.65],
        }
    ).to_csv(bins_csv, index=False)

    plot_reliability_diagram(bins_csv, out_path=out_path, time_window=(0.0, 0.2))

    assert out_path.exists()
    assert out_path.stat().st_size > 0
