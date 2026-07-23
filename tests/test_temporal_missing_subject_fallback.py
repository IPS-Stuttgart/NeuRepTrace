from pathlib import Path

import numpy as np
import pandas as pd

from neureptrace.temporal_model import read_probability_observations


def _write_observation(path: Path) -> None:
    pd.DataFrame(
        {
            "subject": [np.nan],
            "time": [0.0],
            "sample_index": [0],
            "prob_class_0": [0.75],
            "prob_class_1": [0.25],
        }
    ).to_csv(path, index=False)


def test_empty_subject_columns_fall_back_to_each_filename_stem(tmp_path: Path):
    first_path = tmp_path / "subject_alpha.csv"
    second_path = tmp_path / "subject_beta.csv"
    _write_observation(first_path)
    _write_observation(second_path)

    observations = read_probability_observations([first_path, second_path])

    assert observations["subject"].tolist() == ["subject_alpha", "subject_beta"]
