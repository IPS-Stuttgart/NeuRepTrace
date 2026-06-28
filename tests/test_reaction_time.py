from __future__ import annotations

import math
from pathlib import Path

import pytest

from neureptrace.behavior.reaction_time import (
    ReactionTimeCsvConfig,
    analyze_metric_reaction_times,
    extract_reaction_times_from_metadata,
    join_reaction_times,
    load_reaction_time_csv,
    reaction_time_rows_from_values,
)


def test_load_reaction_time_csv_converts_one_based_trials(tmp_path: Path):
    csv_path = tmp_path / "rt.csv"
    csv_path.write_text(
        "participant,dataset,trial,rt\n"
        "2,main,1,0.41\n"
        "2,main,2,0.39\n",
        encoding="utf-8",
    )

    rows = load_reaction_time_csv(csv_path, ReactionTimeCsvConfig(trial_index_base=1))

    assert rows == [
        {"participant": "2", "dataset": "main", "trial": 0, "reaction_time": 0.41},
        {"participant": "2", "dataset": "main", "trial": 1, "reaction_time": 0.39},
    ]


@pytest.mark.parametrize("trial_index_base", [False, True])
def test_load_reaction_time_csv_rejects_boolean_trial_index_base(tmp_path: Path, trial_index_base: bool):
    csv_path = tmp_path / "rt.csv"
    csv_path.write_text(
        "participant,dataset,trial,rt\n"
        "2,main,1,0.41\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="trial_index_base must be one of"):
        load_reaction_time_csv(csv_path, ReactionTimeCsvConfig(trial_index_base=trial_index_base))


@pytest.mark.parametrize("reaction_time_scale", [False, True, math.nan, math.inf])
def test_reaction_time_helpers_reject_invalid_scale(tmp_path: Path, reaction_time_scale: float | bool):
    csv_path = tmp_path / "rt.csv"
    csv_path.write_text(
        "participant,dataset,trial,rt\n"
        "2,main,0,0.41\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reaction_time_scale must be a finite numeric scale"):
        load_reaction_time_csv(csv_path, ReactionTimeCsvConfig(reaction_time_scale=reaction_time_scale))
    with pytest.raises(ValueError, match="reaction_time_scale must be a finite numeric scale"):
        reaction_time_rows_from_values([0.1, 0.2], reaction_time_scale=reaction_time_scale)
    with pytest.raises(ValueError, match="reaction_time_scale must be a finite numeric scale"):
        extract_reaction_times_from_metadata({"rt": [0.1, 0.2]}, reaction_time_scale=reaction_time_scale)


def test_reaction_time_helpers_treat_pandas_missing_scalars_as_nan():
    pd = pytest.importorskip("pandas")

    direct_rows = reaction_time_rows_from_values([0.1, pd.NA, ""], participant="A")
    metadata_rows = extract_reaction_times_from_metadata(pd.DataFrame({"rt": [0.2, pd.NA, ""]}), participant="B")

    assert direct_rows[0]["reaction_time"] == 0.1
    assert math.isnan(direct_rows[1]["reaction_time"])
    assert math.isnan(direct_rows[2]["reaction_time"])
    assert metadata_rows[0]["reaction_time"] == 0.2
    assert math.isnan(metadata_rows[1]["reaction_time"])
    assert math.isnan(metadata_rows[2]["reaction_time"])


@pytest.mark.parametrize("trial_value", ["1.5", "nan", "inf", ""])
def test_load_reaction_time_csv_rejects_invalid_trial_values(tmp_path: Path, trial_value: str):
    csv_path = tmp_path / "rt.csv"
    csv_path.write_text(
        "participant,dataset,trial,rt\n"
        f"2,main,{trial_value},0.41\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="trial values must be finite integers"):
        load_reaction_time_csv(csv_path)


def test_join_reaction_times_rejects_invalid_trial_values():
    metric_rows = [
        {"participant": "2", "dataset": "main", "trial": 0, "score": 1.0},
    ]
    reaction_time_rows = [
        {"participant": "2", "dataset": "main", "trial": 0.5, "reaction_time": 0.4},
    ]

    with pytest.raises(ValueError, match="trial values must be finite integers"):
        join_reaction_times(metric_rows, reaction_time_rows)


def test_join_reaction_times_detects_likely_one_based_trials():
    metric_rows = [
        {"participant": "2", "dataset": "main", "trial": 0, "score": 1.0},
        {"participant": "2", "dataset": "main", "trial": 1, "score": 2.0},
    ]
    reaction_time_rows = [
        {"participant": "2", "dataset": "main", "trial": 1, "reaction_time": 0.4},
        {"participant": "2", "dataset": "main", "trial": 2, "reaction_time": 0.5},
    ]

    with pytest.raises(ValueError, match="look one-based"):
        join_reaction_times(metric_rows, reaction_time_rows)


def test_extract_and_analyze_metric_reaction_times():
    reaction_time_rows = extract_reaction_times_from_metadata(
        {"rt": [0.1, 0.2, 0.3, 0.4]},
        participant="A",
    )
    metric_rows = [
        {"participant": "A", "dataset": "main", "trial": 0, "metric": 1.0},
        {"participant": "A", "dataset": "main", "trial": 1, "metric": 2.0},
        {"participant": "A", "dataset": "main", "trial": 2, "metric": 3.0},
        {"participant": "A", "dataset": "main", "trial": 3, "metric": 4.0},
    ]

    joined = join_reaction_times(metric_rows, reaction_time_rows)
    summary = analyze_metric_reaction_times(joined, metrics=("metric",), min_trials=3)

    participant_row = next(row for row in summary if row["scope"] == "participant")
    assert participant_row["participant"] == "A"
    assert math.isclose(participant_row["pearson_r"], 1.0)
    assert math.isclose(participant_row["slope_reaction_time_per_unit"], 0.1)
