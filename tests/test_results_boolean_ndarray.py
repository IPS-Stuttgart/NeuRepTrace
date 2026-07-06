from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.results import _probability_ece_by_group, aggregate_time_decode_results


def _result_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": ["s1", "s1"],
            "fold": [0, 1],
            "time": [0.1, 0.1],
            "accuracy": [0.6, 0.8],
            "log_loss": [0.5, 0.4],
            "brier": [0.3, 0.2],
            "ece": [0.1, 0.2],
        }
    )


@pytest.mark.parametrize("bad_value", [np.asarray(True), np.asarray(False, dtype=object)])
def test_aggregate_time_decode_results_rejects_boolean_ndarray_metrics(bad_value: object) -> None:
    results = _result_frame()
    results["accuracy"] = results["accuracy"].astype(object)
    results.loc[0, "accuracy"] = bad_value

    with pytest.raises(ValueError, match="Metric column 'accuracy' must not contain booleans"):
        aggregate_time_decode_results(results)


@pytest.mark.parametrize("bad_value", [np.asarray(True), np.asarray(False, dtype=object)])
def test_probability_ece_by_group_rejects_boolean_ndarray_probabilities(bad_value: object) -> None:
    observations = pd.DataFrame(
        {
            "subject": ["s1", "s1"],
            "time": [0.1, 0.1],
            "true_label": [0, 1],
            "prob_class_0": [0.8, 0.2],
            "prob_class_1": [0.2, 0.8],
        }
    )
    observations["prob_class_0"] = observations["prob_class_0"].astype(object)
    observations.loc[0, "prob_class_0"] = bad_value

    with pytest.raises(ValueError, match="Probability-observation column 'prob_class_0' must not contain booleans"):
        _probability_ece_by_group(observations, ["subject", "time"], n_bins=10)
