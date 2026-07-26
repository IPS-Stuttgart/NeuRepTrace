from __future__ import annotations

import pytest

from neureptrace.behavior.reaction_time import analyze_metric_reaction_times


def test_analyze_metric_reaction_times_treats_string_as_single_metric() -> None:
    rows = [
        {"participant": "A", "metric": 1.0, "reaction_time": 0.1},
        {"participant": "A", "metric": 2.0, "reaction_time": 0.2},
        {"participant": "A", "metric": 3.0, "reaction_time": 0.3},
    ]

    summary = analyze_metric_reaction_times(rows, metrics="metric", min_trials=3)

    assert len(summary) == 2
    assert {row["scope"] for row in summary} == {
        "participant",
        "pooled_within_participant",
    }
    assert {row["metric"] for row in summary} == {"metric"}
    assert all(row["pearson_r"] == pytest.approx(1.0) for row in summary)
