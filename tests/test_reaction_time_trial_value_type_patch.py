from __future__ import annotations

import pytest

from neureptrace.behavior.reaction_time import join_reaction_times, reaction_time_rows_from_values


class BadTrialValue:
    def __str__(self) -> str:
        raise TypeError("cannot stringify trial")


def test_reaction_time_rows_from_values_materializes_one_pass_iterables() -> None:
    rows = reaction_time_rows_from_values((value for value in ("0.1", "0.2")), participant="A")

    assert rows == [
        {"participant": "A", "dataset": "main", "trial": 0, "reaction_time": 0.1},
        {"participant": "A", "dataset": "main", "trial": 1, "reaction_time": 0.2},
    ]


def test_join_reaction_times_rejects_unstringifiable_trial_values() -> None:
    metric_rows = [
        {"participant": "2", "dataset": "main", "trial": 0, "score": 1.0},
    ]
    reaction_time_rows = [
        {"participant": "2", "dataset": "main", "trial": BadTrialValue(), "reaction_time": 0.4},
    ]

    with pytest.raises(ValueError, match="trial values must be finite integers"):
        join_reaction_times(metric_rows, reaction_time_rows)
