from __future__ import annotations

import pandas as pd
import pytest

from neureptrace.results import peak_metric_rows, summarize_metric_table


def _mixed_group_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group": pd.Series([1, 1, "1", "1"], dtype=object),
            "time": [-0.2, 0.1, -0.2, 0.1],
            "score": [0.4, 0.8, 0.9, 0.7],
        }
    )


def _rows_by_exact_group(frame: pd.DataFrame) -> dict[tuple[type, object], pd.Series]:
    return {(type(row["group"]), row["group"]): row for _, row in frame.iterrows()}


def test_summarize_metric_table_preserves_mixed_type_group_identifiers() -> None:
    summary = summarize_metric_table(_mixed_group_frame(), "score", "group")

    rows = _rows_by_exact_group(summary)
    assert set(rows) == {(int, 1), (str, "1")}
    assert rows[(int, 1)]["n_rows"] == 2
    assert rows[(int, 1)]["score_mean"] == pytest.approx(0.6)
    assert rows[(str, "1")]["n_rows"] == 2
    assert rows[(str, "1")]["score_mean"] == pytest.approx(0.8)


def test_peak_metric_rows_preserves_mixed_type_group_identifiers() -> None:
    peaks = peak_metric_rows(_mixed_group_frame(), "score", ("group",))

    rows = _rows_by_exact_group(peaks)
    assert set(rows) == {(int, 1), (str, "1")}
    assert rows[(int, 1)]["time"] == pytest.approx(0.1)
    assert rows[(int, 1)]["score"] == pytest.approx(0.8)
    assert rows[(str, "1")]["time"] == pytest.approx(-0.2)
    assert rows[(str, "1")]["score"] == pytest.approx(0.9)
