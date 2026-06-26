from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neureptrace.calibration import aggregate_reliability_bins, build_calibration_report, summarize_calibration_metrics


def _summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decoder": ["logistic", "logistic", "lda", "lda"],
            "time": [-0.05, 0.15, -0.05, 0.15],
            "accuracy_mean": [0.50, 0.60, 0.54, 0.61],
            "log_loss_mean": [0.70, 0.66, 0.78, 0.74],
            "brier_mean": [0.50, 0.47, 0.55, 0.52],
            "ece_mean": [0.09, 0.06, 0.15, 0.12],
            "n_subjects": [5, 5, 5, 5],
        }
    )


def test_summarize_calibration_metrics_orders_by_effect_ece():
    summary = summarize_calibration_metrics(
        _summary_frame(),
        baseline_window=(-0.1, 0.0),
        effect_window=(0.1, 0.2),
    )

    assert summary["decoder"].tolist() == ["logistic", "lda"]
    assert summary["effect_ece_mean"].round(3).tolist() == [0.06, 0.12]
    assert summary["best_ece_time"].tolist() == [0.15, 0.15]


@pytest.mark.parametrize(
    "bad_window",
    [
        True,
        "0.1,0.2",
        (False, 0.2),
        (0.1, np.bool_(True)),
        (0.1, np.inf),
        (0.2, 0.1),
        (0.1,),
        (0.1, 0.2, 0.3),
    ],
)
def test_summarize_calibration_metrics_rejects_malformed_time_windows(bad_window):
    with pytest.raises(ValueError, match="effect_window"):
        summarize_calibration_metrics(_summary_frame(), effect_window=bad_window)


def test_build_calibration_report_normalizes_numeric_window_endpoints(tmp_path: Path):
    summary_csv = tmp_path / "summary.csv"
    _summary_frame().to_csv(summary_csv, index=False)

    report = build_calibration_report(summary_csv, effect_window=("0.1", "0.2"))

    assert "- Effect window: 0.100 to 0.200 s" in report


def test_summarize_calibration_metrics_rejects_non_finite_values():
    frame = _summary_frame()
    frame.loc[0, "log_loss_mean"] = np.inf

    with pytest.raises(ValueError, match="non-finite values.*log_loss_mean"):
        summarize_calibration_metrics(frame)


def test_summarize_calibration_metrics_rejects_out_of_range_rates():
    frame = _summary_frame()
    frame.loc[0, "ece_mean"] = 1.2

    with pytest.raises(ValueError, match=r"outside \[0, 1\].*ece_mean"):
        summarize_calibration_metrics(frame)


def test_summarize_calibration_metrics_rejects_malformed_subject_counts():
    frame = _summary_frame()
    frame["n_subjects"] = frame["n_subjects"].astype(float)
    frame.loc[0, "n_subjects"] = 4.5

    with pytest.raises(ValueError, match="non-integer n_subjects"):
        summarize_calibration_metrics(frame)


def test_aggregate_reliability_bins_weights_by_samples(tmp_path: Path):
    first = tmp_path / "sub-01_calibration_bins.csv"
    second = tmp_path / "sub-02_calibration_bins.csv"
    pd.DataFrame(
        {
            "decoder": ["logistic"],
            "time": [0.1],
            "bin": [5],
            "bin_left": [0.5],
            "bin_right": [0.6],
            "n_samples": [10],
            "accuracy": [0.8],
            "confidence": [0.6],
        }
    ).to_csv(first, index=False)
    pd.DataFrame(
        {
            "decoder": ["logistic"],
            "time": [0.1],
            "bin": [5],
            "bin_left": [0.5],
            "bin_right": [0.6],
            "n_samples": [30],
            "accuracy": [0.4],
            "confidence": [0.5],
        }
    ).to_csv(second, index=False)

    aggregated = aggregate_reliability_bins([first, second])

    assert aggregated["n_samples"].tolist() == [40]
    assert aggregated["accuracy"].round(3).tolist() == [0.5]
    assert aggregated["confidence"].round(3).tolist() == [0.525]
    assert aggregated["gap"].round(3).tolist() == [-0.025]


def test_aggregate_reliability_bins_accepts_empty_bins_from_reliability_output(tmp_path: Path):
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
    assert pd.isna(aggregated.loc[0, "accuracy"])
    assert pd.isna(aggregated.loc[0, "confidence"])
    assert pd.isna(aggregated.loc[0, "gap"])


def test_aggregate_reliability_bins_rejects_malformed_probability_bins(tmp_path: Path):
    path = tmp_path / "bad_calibration_bins.csv"
    pd.DataFrame(
        {
            "time": [0.1],
            "bin": [5],
            "bin_left": [0.5],
            "bin_right": [0.6],
            "n_samples": [10],
            "accuracy": [1.2],
            "confidence": [0.6],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="outside \\[0, 1\\].*accuracy"):
        aggregate_reliability_bins([path])


def test_aggregate_reliability_bins_rejects_missing_positive_bin_values(tmp_path: Path):
    path = tmp_path / "missing_positive_calibration_bins.csv"
    pd.DataFrame(
        {
            "time": [0.1],
            "bin": [5],
            "bin_left": [0.5],
            "bin_right": [0.6],
            "n_samples": [10],
            "accuracy": [np.nan],
            "confidence": [0.6],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing or non-finite values.*accuracy.*positive n_samples"):
        aggregate_reliability_bins([path])


def test_aggregate_reliability_bins_rejects_non_integer_sample_counts(tmp_path: Path):
    path = tmp_path / "bad_sample_counts.csv"
    pd.DataFrame(
        {
            "time": [0.1],
            "bin": [5],
            "bin_left": [0.5],
            "bin_right": [0.6],
            "n_samples": [10.5],
            "accuracy": [0.8],
            "confidence": [0.6],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="non-integer n_samples"):
        aggregate_reliability_bins([path])


def test_build_calibration_report_writes_markdown(tmp_path: Path):
    summary_csv = tmp_path / "summary.csv"
    _summary_frame().to_csv(summary_csv, index=False)

    report = build_calibration_report(summary_csv, effect_window=(0.1, 0.2))

    assert "# NeuRepTrace Calibration Report" in report
    assert "| logistic | 5 | 0.060 | 0.470 | 0.660 | 0.600 |" in report


def test_build_calibration_report_defaults_decoder_for_emission_only_summary(tmp_path: Path):
    summary_csv = tmp_path / "summary.csv"
    frame = _summary_frame().drop(columns="decoder")
    frame["emission_mode"] = "calibrated"
    frame.to_csv(summary_csv, index=False)

    report = build_calibration_report(summary_csv, effect_window=(0.1, 0.2))

    assert "| Decoder | Emission mode |" in report
    assert "| overall | calibrated | 5 | 0.090 | 0.495 | 0.700 | 0.605 |" in report
