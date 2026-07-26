from __future__ import annotations

import pandas as pd

from neureptrace.metrics import compare_prepost_windows, summarize_window_metric


def _categorical_group_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "condition": pd.Categorical(
                ["observed", "observed"],
                categories=["observed", "unused"],
            ),
            "time": [-0.1, 0.1],
            "score": [1.0, 3.0],
        }
    )


def test_window_summary_ignores_unobserved_categorical_group_levels() -> None:
    summary = summarize_window_metric(
        _categorical_group_frame(),
        "score",
        (-0.1, 0.1),
        group_columns=("condition",),
    )

    assert summary["condition"].tolist() == ["observed"]
    assert summary["n_rows"].tolist() == [2]


def test_prepost_comparison_ignores_unobserved_categorical_group_levels() -> None:
    comparison = compare_prepost_windows(
        _categorical_group_frame(),
        "score",
        (-0.1, -0.1),
        (0.1, 0.1),
        group_columns=("condition",),
    )

    assert comparison["condition"].tolist() == ["observed"]
    assert comparison["score_post_minus_pre"].tolist() == [2.0]
