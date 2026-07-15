from __future__ import annotations

import math
from pathlib import Path

import pytest

from neureptrace.behavior.reaction_time import (
    ReactionTimeCsvConfig,
    join_reaction_times,
    load_reaction_time_csv,
    reaction_time_rows_from_values,
)

_HUGE_INTEGER = 10**10_000


def _write_reaction_time_csv(path: Path) -> None:
    path.write_text(
        "participant,dataset,trial,rt\n"
        "2,main,0,0.41\n",
        encoding="utf-8",
    )


def test_reaction_time_scale_overflow_raises_controlled_value_error(tmp_path: Path):
    csv_path = tmp_path / "rt.csv"
    _write_reaction_time_csv(csv_path)

    with pytest.raises(ValueError, match="reaction_time_scale must be a finite numeric scale"):
        load_reaction_time_csv(csv_path, ReactionTimeCsvConfig(reaction_time_scale=_HUGE_INTEGER))
    with pytest.raises(ValueError, match="reaction_time_scale must be a finite numeric scale"):
        reaction_time_rows_from_values([0.41], reaction_time_scale=_HUGE_INTEGER)


def test_reaction_time_trial_index_base_error_survives_huge_integer_repr(tmp_path: Path):
    csv_path = tmp_path / "rt.csv"
    _write_reaction_time_csv(csv_path)

    with pytest.raises(ValueError, match="trial_index_base must be one of"):
        load_reaction_time_csv(csv_path, ReactionTimeCsvConfig(trial_index_base=_HUGE_INTEGER))


def test_overflowing_reaction_time_values_are_treated_as_missing():
    rows = reaction_time_rows_from_values([_HUGE_INTEGER])

    assert math.isnan(rows[0]["reaction_time"])


def test_huge_programmatic_trial_values_raise_controlled_validation_errors():
    metric_rows = [
        {"participant": "2", "dataset": "main", "trial": _HUGE_INTEGER, "score": 1.0},
    ]
    reaction_time_rows = [
        {"participant": "2", "dataset": "main", "trial": 0, "reaction_time": 0.41},
    ]

    with pytest.raises(ValueError, match="trial values must be finite integers"):
        join_reaction_times(metric_rows, reaction_time_rows, detect_one_based_trials=False)
