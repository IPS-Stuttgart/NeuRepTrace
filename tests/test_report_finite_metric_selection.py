import pandas as pd
import pytest

from neureptrace.report import summarize_aggregate_time_decode


def _summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [-0.05, 0.15, 0.25],
            "accuracy_mean": [0.49, 0.61, 0.58],
            "accuracy_sem": [0.01, 0.02, 0.03],
            "log_loss_mean": [0.72, 0.65, 0.68],
            "brier_mean": [0.52, 0.45, 0.48],
            "ece_mean": [0.11, 0.07, 0.08],
            "n_subjects": [2, 2, 2],
        }
    )


def test_report_selection_ignores_nonfinite_metric_candidates() -> None:
    frame = _summary_frame()
    frame.loc[1, "brier_mean"] = -1e309

    summary = summarize_aggregate_time_decode(
        frame,
        baseline_window=(-0.1, 0.0),
        effect_window=(0.1, 0.3),
        selection_metric="brier",
    )

    assert summary["selected_time"] == 0.25
    assert summary["selected_score"] == 0.48
    assert summary["effect_selection_mean"] == 0.48
    assert round(summary["selection_improvement"], 3) == 0.04


def test_report_selection_rejects_all_nonfinite_metric_values() -> None:
    frame = _summary_frame()
    frame["brier_mean"] = [float("nan"), -1e309, 1e309]

    with pytest.raises(ValueError, match="contains no finite values"):
        summarize_aggregate_time_decode(
            frame,
            baseline_window=(-0.1, 0.0),
            effect_window=(0.1, 0.3),
            selection_metric="brier",
        )
