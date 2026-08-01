from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neureptrace.semantic_stages import read_state_traces


def _write_state_trace(path: Path, subject: object = np.nan) -> None:
    pd.DataFrame(
        {
            "subject": [subject],
            "time": [0.0],
            "viterbi_class": ["state"],
            "state_0": ["state"],
            "posterior_state_0": [1.0],
        }
    ).to_csv(path, index=False)


def test_empty_subject_columns_fall_back_to_each_state_trace_filename(tmp_path: Path) -> None:
    first_path = tmp_path / "subject_alpha.csv"
    second_path = tmp_path / "subject_beta.csv"
    _write_state_trace(first_path)
    _write_state_trace(second_path)

    traces = read_state_traces([first_path, second_path])

    assert traces["subject"].tolist() == ["subject_alpha", "subject_beta"]


@pytest.mark.parametrize("missing_subject", ["   ", "none", "NaT", " NONE ", " nat "])
def test_textual_missing_subjects_fall_back_to_state_trace_filename(
    tmp_path: Path,
    missing_subject: str,
) -> None:
    trace_path = tmp_path / "subject_gamma.csv"
    _write_state_trace(trace_path, missing_subject)

    traces = read_state_traces([trace_path])

    assert traces["subject"].tolist() == ["subject_gamma"]
