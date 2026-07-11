from pathlib import Path

import pandas as pd

from neureptrace.calibration import build_calibration_report


def test_calibration_report_escapes_group_labels_in_markdown_table(tmp_path: Path) -> None:
    summary_csv = tmp_path / "summary.csv"
    pd.DataFrame(
        {
            "decoder": ["log|istic\nmodel", "log|istic\nmodel"],
            "emission_mode": ["calibrated|held\nout", "calibrated|held\nout"],
            "time": [-0.05, 0.15],
            "accuracy_mean": [0.50, 0.60],
            "log_loss_mean": [0.70, 0.66],
            "brier_mean": [0.50, 0.47],
            "ece_mean": [0.09, 0.06],
            "n_subjects": [5, 5],
        }
    ).to_csv(summary_csv, index=False)

    report = build_calibration_report(summary_csv, effect_window=(0.1, 0.2))

    assert "| log\\|istic model | calibrated\\|held out | 5 |" in report
    assert "| log|istic" not in report
    assert "calibrated|held" not in report
