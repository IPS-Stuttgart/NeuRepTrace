from pathlib import Path

import pandas as pd
import pytest

from neureptrace.temporal_model import fit_temporal_models


def _observation_frame(*, class_0: str = "left", class_1: str = "right", subject: str = "sub-01") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject": subject,
                "decoder": "logistic",
                "emission_mode": "calibrated",
                "sequence_id": 0,
                "time": 0.10,
                "class_0": class_0,
                "class_1": class_1,
                "prob_class_0": 0.9,
                "prob_class_1": 0.1,
            },
            {
                "subject": subject,
                "decoder": "logistic",
                "emission_mode": "calibrated",
                "sequence_id": 0,
                "time": 0.20,
                "class_0": class_0,
                "class_1": class_1,
                "prob_class_0": 0.8,
                "prob_class_1": 0.2,
            },
        ]
    )


def _fit(paths: list[Path]) -> None:
    fit_temporal_models(
        paths,
        effect_window=(0.10, 0.20),
        baseline_window=None,
        n_permutations=0,
        stay_grid_size=5,
    )


def test_fit_temporal_models_rejects_inconsistent_class_mapping_across_files(tmp_path: Path) -> None:
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    _observation_frame().to_csv(first_path, index=False)
    _observation_frame(class_0="right", class_1="left", subject="sub-02").to_csv(second_path, index=False)

    with pytest.raises(ValueError, match="class_0 maps to multiple classes"):
        _fit([first_path, second_path])


def test_fit_temporal_models_rejects_duplicate_class_names(tmp_path: Path) -> None:
    csv_path = tmp_path / "duplicate_class_names.csv"
    _observation_frame(class_0="left", class_1="left").to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="map probability columns to distinct classes"):
        _fit([csv_path])
