from __future__ import annotations

import pandas as pd

from neureptrace.bushmeg_all_protocols_report import build_protocol3_kshot_leaderboard


def test_protocol3_kshot_leaderboard_counts_repeated_subject_rows_once_per_row() -> None:
    summary = pd.DataFrame([
        {"protocol_category": 1, "method": "source_loso_logistic", "method_family": "source", "outer_test_subject": "1", "balanced_accuracy": 0.50},
        {"protocol_category": 3, "method": "few_shot_decoder_k2", "method_family": "few_shot", "outer_test_subject": "1", "balanced_accuracy": 0.60, "k_per_class": 2, "n_target_evaluation_trials": 10, "n_target_calibration_trials": 4},
        {"protocol_category": 3, "method": "few_shot_decoder_k2", "method_family": "few_shot", "outer_test_subject": "1", "balanced_accuracy": 0.70, "k_per_class": 2, "n_target_evaluation_trials": 11, "n_target_calibration_trials": 4},
    ])

    row = build_protocol3_kshot_leaderboard(summary).iloc[0]

    assert len(build_protocol3_kshot_leaderboard(summary)) == 1
    assert round(float(row["mean_balanced_accuracy"]), 10) == 0.65
    assert round(float(row["mean_delta_vs_source_loso_logistic"]), 10) == 0.15
    assert int(row["n_eval_trials"]) == 21
    assert int(row["n_calibration_trials"]) == 8
