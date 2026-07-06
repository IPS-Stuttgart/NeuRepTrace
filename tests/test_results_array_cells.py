from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.results import _probability_ece_by_group, aggregate_time_decode_results


def _result_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": ["s1", "s1", "s1", "s1"],
            "fold": [0, 1, 0, 1],
            "time": [0.1, 0.1, 0.2, 0.2],
            "accuracy": [0.6, 0.8, 0.7, 0.9],
            "log_loss": [0.5, 0.4, 0.45, 0.35],
            "brier": [0.3, 0.2, 0.25, 0.15],
            "ece": [0.1, 0.2, 0.15, 0.25],
        }
    )


@pytest.mark.parametrize("bad_value", [np.asarray(True), np.asarray([True]), np.asarray(1), np.asarray([1])])
def test_aggregate_time_decode_results_rejects_array_metric_cells(bad_value: object) -> None:
    results = _result_frame()
    results["accuracy"] = results["accuracy"].astype(object)
    results.at[0, "accuracy"] = bad_value

    with pytest.raises(ValueError, match="Metric column 'accuracy'.*arrays"):
        aggregate_time_decode_results(results)


@pytest.mark.parametrize("bad_value", [np.asarray(True), np.asarray([True]), np.asarray(1), np.asarray([1])])
def test_aggregate_time_decode_results_rejects_array_n_test_cells(bad_value: object) -> None:
    results = _result_frame()
    results["n_test"] = pd.Series([3, 3, 3, 3], dtype=object)
    results.at[0, "n_test"] = bad_value

    with pytest.raises(ValueError, match="positive integer fold sizes"):
        aggregate_time_decode_results(results)


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [
        ("time", np.asarray(0.1)),
        ("true_label", np.asarray(0)),
        ("prob_class_0", np.asarray([0.8])),
    ],
)
def test_probability_ece_by_group_rejects_array_cells(column: str, bad_value: object) -> None:
    observations = pd.DataFrame(
        {
            "subject": ["s1", "s1"],
            "time": [0.184, 0.184],
            "true_label": [0, 1],
            "prob_class_0": [0.8, 0.2],
            "prob_class_1": [0.2, 0.8],
        }
    )
    observations[column] = observations[column].astype(object)
    observations.at[0, column] = bad_value

    with pytest.raises(ValueError, match=f"Probability-observation column '{column}'.*arrays"):
        _probability_ece_by_group(observations, ["subject", "time"], n_bins=10)
