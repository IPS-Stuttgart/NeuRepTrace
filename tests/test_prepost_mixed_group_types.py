from __future__ import annotations

import pandas as pd

from neureptrace.metrics import compare_prepost_windows, summarize_window_metric


def _mixed_group_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group": pd.Series([1, 1, "1", "1"], dtype=object),
            "time": [-0.1, 0.1, -0.1, 0.1],
            "score": [1.0, 3.0, 10.0, 14.0],
        }
    )


def _rows_by_exact_group(frame: pd.DataFrame) -> dict[tuple[type, object], pd.Series]:
    return {(type(row["group"]), row["group"]): row for _, row in frame.iterrows()}


def test_prepost_metrics_preserve_mixed_type_group_identifiers() -> None:
    frame = _mixed_group_frame()

    summary = summarize_window_metric(frame, "score", (-0.1, 0.1), group_columns=("group",))
    comparison = compare_prepost_windows(frame, "score", (-0.1, -0.1), (0.1, 0.1), group_columns=("group",))

    summary_rows = _rows_by_exact_group(summary)
    assert set(summary_rows) == {(int, 1), (str, "1")}
    assert summary_rows[(int, 1)]["score_mean"] == 2.0
    assert summary_rows[(str, "1")]["score_mean"] == 12.0

    comparison_rows = _rows_by_exact_group(comparison)
    assert set(comparison_rows) == {(int, 1), (str, "1")}
    assert comparison_rows[(int, 1)]["score_post_minus_pre"] == 2.0
    assert comparison_rows[(str, "1")]["score_post_minus_pre"] == 4.0
