from __future__ import annotations

import pandas as pd
import pytest

from neureptrace.bushmeg_all_protocols_report import build_protocol3_kshot_leaderboard


def test_protocol3_kshot_leaderboard_does_not_multiply_duplicate_subject_rows() -> None:
    summary = pd.DataFrame(
        [
            {
                "protocol_category": 1,
                "method": "source_loso_logistic",
                "method_family": "source",
                "outer_test_subject": "1",
                "balanced_accuracy": 0.50,
            },
            {
                "protocol_category": 3,
                "method": "few_shot_decoder_k2",
                "method_family": "few_shot",
                "outer_test_subject": "1",
                "feature_kind": "sensor",
                "balanced_accuracy": 0.60,
                "k_per_class": 2,
                "n_target_evaluation_trials": 30,
                "n_target_calibration_trials": 6,
            },
            {
                "protocol_category": 3,
                "method": "few_shot_decoder_k2",
                "method_family": "few_shot",
                "outer_test_subject": "1",
                "feature_kind": "source",
                "balanced_accuracy": 0.80,
                "k_per_class": 2,
                "n_target_evaluation_trials": 30,
                "n_target_calibration_trials": 6,
            },
        ]
    )

    leaderboard = build_protocol3_kshot_leaderboard(summary)

    assert len(leaderboard) == 1
    row = leaderboard.iloc[0]
    assert row["mean_balanced_accuracy"] == pytest.approx(0.70)
    assert row["mean_delta_vs_source_loso_logistic"] == pytest.approx(0.20)
    assert int(row["n_subjects"]) == 1
    assert int(row["n_eval_trials"]) == 60
    assert int(row["n_calibration_trials"]) == 12
