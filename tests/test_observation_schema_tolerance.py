from __future__ import annotations

import pandas as pd
import pytest

from neureptrace.observation_schema import validate_probability_observations
from neureptrace.observation_schema import main as validate_observations_main


def _valid_observations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": ["sub-01", "sub-01"],
            "sequence_id": ["trial-001", "trial-001"],
            "time": [-0.1, -0.08],
            "decoder": ["logistic", "logistic"],
            "emission_mode": ["calibrated", "calibrated"],
            "prob_class_0": [0.6, 0.45],
            "prob_class_1": [0.4, 0.55],
        }
    )


@pytest.mark.parametrize("probability_tolerance", [float("nan"), float("inf"), float("-inf"), -1e-6])
def test_invalid_probability_tolerance_is_rejected(probability_tolerance: float) -> None:
    with pytest.raises(ValueError, match="probability_tolerance must be finite and non-negative"):
        validate_probability_observations(_valid_observations(), probability_tolerance=probability_tolerance)


def test_validate_observations_cli_rejects_non_finite_probability_tolerance(tmp_path) -> None:
    csv_path = tmp_path / "observations.csv"
    _valid_observations().to_csv(csv_path, index=False)

    exit_code = validate_observations_main([str(csv_path), "--probability-tolerance", "nan"])

    assert exit_code == 2
