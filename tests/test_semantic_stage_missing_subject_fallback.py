from pathlib import Path

import numpy as np
import pandas as pd

from neureptrace.semantic_stages import read_state_traces


def _write_state_trace(path: Path) -> None:
    pd.DataFrame(
        {
            "subject": [np.nan],
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
