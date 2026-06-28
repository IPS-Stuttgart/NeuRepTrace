from __future__ import annotations

import pandas as pd

import neureptrace  # noqa: F401
from neureptrace import bushmeg_all_protocols_report as report


def test_fractional_protocol_report_patch_installed_on_package_import() -> None:
    summary = pd.DataFrame(
        {
            "protocol_category": [1.5],
            "method": ["fractional_protocol_method"],
            "method_family": ["smoke"],
            "outer_test_subject": ["sub-01"],
            "balanced_accuracy": [0.75],
            "accuracy": [0.80],
            "log_loss": [0.20],
            "brier": [0.10],
            "ece": [0.05],
        }
    )
    leaderboard = pd.DataFrame(
        {
            "protocol_category": [1.5],
            "method": ["fractional_protocol_method"],
            "n_rows": [1],
            "n_skipped": [0],
        }
    )

    protocol_summary = report.build_protocol_summary(summary, leaderboard)

    assert protocol_summary.loc[0, "protocol_category"] == 1.5
    assert protocol_summary.loc[0, "n_rows"] == 1
    assert report._format_protocol_label(1.5) == "1.5"
