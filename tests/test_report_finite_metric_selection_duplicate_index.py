from __future__ import annotations

import pandas as pd

from neureptrace.report import summarize_aggregate_time_decode


def _summary(index: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [-0.05, 0.15, 0.25],
            "accuracy_mean": [0.90, 0.60, 0.50],
            "accuracy_sem": [0.01, 0.02, 0.03],
            "log_loss_mean": [0.72, 0.65, 0.68],
            "brier_mean": [0.60, 0.20, 0.30],
            "ece_mean": [0.11, 0.07, 0.08],
            "n_subjects": [2, 2, 2],
        },
        index=index,
    )


def test_metric_selection_uses_position_with_duplicate_index_labels() -> None:
    result = summarize_aggregate_time_decode(
        _summary([10, 0, 0]),
        baseline_window=(-0.1, 0.0),
        effect_window=(0.1, 0.3),
        selection_metric="brier",
    )

    assert result["peak_time"] == -0.05
    assert result["selected_time"] == 0.15
    assert result["selected_score"] == 0.20


def test_peak_selection_uses_position_with_duplicate_index_labels() -> None:
    result = summarize_aggregate_time_decode(
        _summary([10, 10, 0]),
        baseline_window=(-0.1, 0.0),
        effect_window=(0.1, 0.3),
        selection_metric="brier",
    )

    assert result["peak_time"] == -0.05
    assert result["peak_accuracy"] == 0.90
    assert result["selected_time"] == 0.15
