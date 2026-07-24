from __future__ import annotations

import math

import numpy as np
import pytest

from neureptrace.behavior.reaction_time import (
    analyze_metric_reaction_times,
    extract_reaction_times_from_metadata,
    reaction_time_rows_from_values,
)


@pytest.mark.parametrize("value", [True, np.bool_(False)])
def test_reaction_time_rows_reject_boolean_observations(value: object) -> None:
    with pytest.raises(ValueError, match="not boolean flags"):
        reaction_time_rows_from_values([0.1, value], participant="A")


def test_metadata_reaction_times_reject_boolean_observations() -> None:
    with pytest.raises(ValueError, match="not boolean flags"):
        extract_reaction_times_from_metadata({"rt": [0.1, True]}, participant="A")


def test_boolean_reaction_times_are_excluded_from_associations() -> None:
    rows = [
        {"participant": "A", "metric": 0.0, "reaction_time": True},
        {"participant": "A", "metric": 1.0, "reaction_time": 0.1},
        {"participant": "A", "metric": 2.0, "reaction_time": 0.2},
    ]

    summary = analyze_metric_reaction_times(rows, metrics=("metric",), min_trials=2)
    participant_row = next(row for row in summary if row["scope"] == "participant")

    assert participant_row["n_trials"] == 2
    assert math.isclose(participant_row["reaction_time_mean"], 0.15)
    assert math.isclose(participant_row["pearson_r"], 1.0)
