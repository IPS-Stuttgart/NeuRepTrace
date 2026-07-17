from __future__ import annotations

import math

import numpy as np

from neureptrace.behavior.reaction_time import analyze_metric_reaction_times


def test_reaction_time_association_preserves_extreme_finite_constant_rows():
    max_float = np.finfo(float).max
    rows = [
        {
            "participant": "A",
            "metric": max_float,
            "reaction_time": max_float,
        }
        for _ in range(3)
    ]

    with np.errstate(over="raise", invalid="raise"):
        summary = analyze_metric_reaction_times(rows, metrics=("metric",), min_trials=3)

    participant_row = next(row for row in summary if row["scope"] == "participant")
    pooled_row = next(row for row in summary if row["scope"] == "pooled_within_participant")

    assert participant_row["n_trials"] == 3
    assert participant_row["metric_mean"] == max_float
    assert participant_row["reaction_time_mean"] == max_float
    assert math.isnan(participant_row["pearson_r"])

    assert pooled_row["n_trials"] == 3
    assert pooled_row["metric_mean"] == 0.0
    assert pooled_row["reaction_time_mean"] == 0.0
    assert math.isnan(pooled_row["pearson_r"])
