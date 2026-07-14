from __future__ import annotations

import pandas as pd

from neureptrace.semantic_stages import build_stage_report, detect_stable_stages


def _duplicate_index_time_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decoder": ["logistic", "logistic", "logistic"],
            "emission_mode": ["calibrated", "calibrated", "calibrated"],
            "true_class": ["animate", "animate", "animate"],
            "time": [0.1, 0.2, 0.3],
            "posterior_true_class_mean": [0.7, 0.9, 0.8],
            "viterbi_match_rate": [0.8, 0.9, 0.85],
            "n_subjects": [2, 2, 2],
            "n_sequences": [4, 4, 4],
        },
        index=[5, 5, 9],
    )


def test_detect_stable_stages_selects_peak_by_position_with_duplicate_indices() -> None:
    stages = detect_stable_stages(
        _duplicate_index_time_summary(),
        posterior_threshold=0.6,
        match_threshold=0.6,
        min_duration=0.1,
    )

    assert len(stages) == 1
    assert stages.loc[0, "peak_time"] == 0.2
    assert stages.loc[0, "peak_posterior_true_class"] == 0.9


def test_build_stage_report_selects_one_peak_per_group_with_duplicate_indices() -> None:
    report = build_stage_report(
        _duplicate_index_time_summary(),
        pd.DataFrame(),
        posterior_threshold=0.6,
        match_threshold=0.6,
        min_duration=0.1,
    )

    peak_row = "| logistic | calibrated | animate | 0.200 | 0.900 | 0.900 |"
    assert report.count("| logistic | calibrated | animate |") == 1
    assert peak_row in report
