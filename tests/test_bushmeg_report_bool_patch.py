from __future__ import annotations

import pandas as pd

from neureptrace.bushmeg_all_protocols_report import build_leaderboard


def test_build_leaderboard_accepts_repeated_boolean_metadata() -> None:
    summary = pd.DataFrame(
        [
            {
                "protocol_category": 1,
                "method": "source_loso_raw",
                "method_family": "source_loso",
                "outer_test_subject": "sub-01",
                "balanced_accuracy": 0.75,
                "accuracy": 0.80,
                "log_loss": 0.50,
                "brier": 0.20,
                "ece": 0.10,
            }
        ]
    )
    metadata = pd.DataFrame(
        [
            {
                "method": "source_loso_raw",
                "method_family": "source_loso",
                "protocol_category": 1,
                "runnable": [True, True],
                "valid_for_zero_calibration": [True, True],
                "valid_for_strict_source_only": [True, True],
                "debug_upper_bound": [False, False],
            }
        ]
    )

    leaderboard = build_leaderboard(summary, metadata)
    row = leaderboard.loc[leaderboard["method"] == "source_loso_raw"].iloc[0]

    assert bool(row["valid_for_zero_calibration"]) is True
    assert bool(row["valid_for_strict_source_only"]) is True
    assert bool(row["debug_upper_bound"]) is False
    assert int(row["n_skipped"]) == 0
