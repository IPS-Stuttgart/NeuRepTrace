from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from neureptrace.observation_schema import validate_probability_observations
from neureptrace.temporal_model import read_probability_observations


def _temporal_observations(sequence_ids: list[object]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": ["sub-01"] * len(sequence_ids),
            "sequence_id": sequence_ids,
            "time": [0.1, 0.2, 0.1, 0.2],
            "decoder": ["logistic"] * len(sequence_ids),
            "emission_mode": ["calibrated"] * len(sequence_ids),
            "prob_class_0": [0.8, 0.7, 0.4, 0.3],
            "prob_class_1": [0.2, 0.3, 0.6, 0.7],
        }
    )


def test_temporal_profile_rejects_missing_sequence_identifier_values() -> None:
    frame = _temporal_observations([None, " ", "trial-002", "trial-002"])

    report = validate_probability_observations(frame, profile="temporal-model")

    identifier_errors = [issue for issue in report.errors if issue.code == "missing_sequence_identifier_value"]
    assert [issue.row for issue in identifier_errors] == [0, 1]
    assert all(issue.column == "sequence_id" for issue in identifier_errors)


def test_temporal_model_reader_rejects_missing_sequence_identifier_values(tmp_path: Path) -> None:
    csv_path = tmp_path / "observations.csv"
    _temporal_observations([None, None, "trial-002", "trial-002"]).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match=r"sequence_id.*missing or blank"):
        read_probability_observations([csv_path])
