from __future__ import annotations

import pandas as pd

from neureptrace.results import summarize_metric_table


def _rows_by_exact_group(frame: pd.DataFrame) -> dict[tuple[type, object], pd.Series]:
    return {(type(row["group"]), row["group"]): row for _, row in frame.iterrows()}


def test_summarize_metric_table_preserves_mixed_type_group_identifiers() -> None:
    frame = pd.DataFrame(
        {
            "group": pd.Series([1, 1, "1", "1"], dtype=object),
            "score": [1.0, 3.0, 10.0, 14.0],
        }
    )

    summary = summarize_metric_table(frame, "score", "group")

    rows = _rows_by_exact_group(summary)
    assert set(rows) == {(int, 1), (str, "1")}
    assert rows[(int, 1)]["n_rows"] == 2
    assert rows[(int, 1)]["score_mean"] == 2.0
    assert rows[(str, "1")]["n_rows"] == 2
    assert rows[(str, "1")]["score_mean"] == 12.0
