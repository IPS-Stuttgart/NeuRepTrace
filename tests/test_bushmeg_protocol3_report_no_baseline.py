from __future__ import annotations

import numpy as np
import pandas as pd

from neureptrace.bushmeg_all_protocols_report import build_protocol3_kshot_leaderboard


def test_protocol3_kshot_leaderboard_keeps_rows_without_protocol1_baseline() -> None:
    summary = pd.DataFrame(
        {
            "protocol_category": [3, 3],
            "method": ["fewshot_lora_k1", "fewshot_lora_k1"],
            "method_family": ["few_shot", "few_shot"],
            "outer_test_subject": ["sub-01", "sub-02"],
            "balanced_accuracy": [0.25, 0.35],
            "target_calibration_per_class": [1, 1],
            "n_target_evaluation_trials": [32, 32],
            "n_target_calibration_trials": [8, 8],
        }
    )

    leaderboard = build_protocol3_kshot_leaderboard(summary)

    assert len(leaderboard) == 1
    row = leaderboard.iloc[0]
    assert row["method"] == "fewshot_lora_k1"
    assert row["method_base"] == "fewshot_lora"
    assert row["method_family"] == "few_shot"
    assert row["k_per_class"] == 1
    assert row["n_subjects"] == 2
    assert row["n_eval_trials"] == 64
    assert row["n_calibration_trials"] == 16
    assert row["mean_balanced_accuracy"] == 0.30
    assert np.isnan(row["mean_delta_vs_source_loso_logistic"])
    assert np.isnan(row["mean_delta_vs_best_protocol1"])
