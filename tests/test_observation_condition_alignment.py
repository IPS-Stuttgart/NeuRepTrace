from pathlib import Path

import pandas as pd

from neureptrace.results import aggregate_time_decode_csvs


def test_observations_inherit_singleton_nondefault_condition_from_csv(tmp_path: Path):
    result_csv = tmp_path / "result.csv"
    observation_csv = tmp_path / "observations.csv"
    summary_csv = tmp_path / "summary.csv"
    pd.DataFrame(
        {
            "subject": ["s1", "s1"],
            "time": [0.1, 0.1],
            "accuracy": [0.6, 0.8],
            "log_loss": [0.5, 0.4],
            "brier": [0.3, 0.2],
            "ece": [0.9, 0.9],
            "emission_mode": ["uncalibrated", "uncalibrated"],
        }
    ).to_csv(result_csv, index=False)
    pd.DataFrame(
        {
            "subject": ["s1", "s1"],
            "time": [0.1, 0.1],
            "true_label": [0, 1],
            "prob_class_0": [0.8, 0.2],
            "prob_class_1": [0.2, 0.8],
        }
    ).to_csv(observation_csv, index=False)

    summary = aggregate_time_decode_csvs([result_csv], summary_csv, observation_csv_paths=[observation_csv])

    assert summary["emission_mode"].tolist() == ["uncalibrated"]
    assert summary["ece_mean"].round(6).tolist() == [0.2]
