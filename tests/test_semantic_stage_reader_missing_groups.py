from __future__ import annotations

from pathlib import Path

import pandas as pd

from neureptrace.semantic_stages import (
    read_state_traces,
    summarize_dominant_timecourse,
)


def test_state_trace_reader_preserves_missing_optional_group_identifiers(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "states.csv"
    pd.DataFrame(
        {
            "subject": ["sub-01"] * 4,
            "decoder": [pd.NA, pd.NA, "logistic", "logistic"],
            "emission_mode": ["calibrated", "calibrated", pd.NA, pd.NA],
            "sequence_id": [0, 0, 1, 1],
            "time": [0.0, 0.1, 0.0, 0.1],
            "viterbi_class": ["left", "left", "right", "right"],
            "posterior_state_0": [0.8, 0.7, 0.4, 0.3],
            "posterior_state_1": [0.2, 0.3, 0.6, 0.7],
            "state_0": ["left"] * 4,
            "state_1": ["right"] * 4,
        }
    ).to_csv(csv_path, index=False)

    traces = read_state_traces([csv_path])

    assert traces.loc[:1, "decoder"].isna().all()
    assert traces.loc[2:, "emission_mode"].isna().all()
    assert not traces["decoder"].eq("nan").any()
    assert not traces["emission_mode"].eq("nan").any()

    summary = summarize_dominant_timecourse(traces)
    assert len(summary) == 4
    assert int(summary["decoder"].isna().sum()) == 2
    assert int(summary["emission_mode"].isna().sum()) == 2
    assert summary.loc[summary["decoder"].isna(), "emission_mode"].eq(
        "calibrated"
    ).all()
    assert summary.loc[summary["emission_mode"].isna(), "decoder"].eq(
        "logistic"
    ).all()
