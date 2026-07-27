from pathlib import Path

import pandas as pd
import pytest

from neureptrace.calibration import build_calibration_report, summarize_calibration_metrics


SUMMARY_COLUMNS = [
    "decoder",
    "time",
    "accuracy_mean",
    "log_loss_mean",
    "brier_mean",
    "ece_mean",
    "n_subjects",
]


def test_summarize_calibration_metrics_rejects_empty_summary():
    summary = pd.DataFrame(columns=SUMMARY_COLUMNS)

    with pytest.raises(ValueError, match="at least one data row"):
        summarize_calibration_metrics(summary)


def test_build_calibration_report_rejects_header_only_summary(tmp_path: Path):
    summary_csv = tmp_path / "empty_summary.csv"
    pd.DataFrame(columns=SUMMARY_COLUMNS).to_csv(summary_csv, index=False)

    with pytest.raises(ValueError, match="at least one data row"):
        build_calibration_report(summary_csv)
