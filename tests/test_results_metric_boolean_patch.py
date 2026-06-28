from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.results import aggregate_time_decode_results


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


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("accuracy", True),
        ("log_loss", np.bool_(False)),
        ("brier", True),
        ("ece", False),
    ],
)
def test_aggregate_time_decode_results_rejects_boolean_metric_values(column: str, value: object) -> None:
    results = _result_frame()
    results[column] = results[column].astype(object)
    results.loc[0, column] = value

    with pytest.raises(ValueError, match=f"Metric column '{column}'.*booleans"):
        aggregate_time_decode_results(results)
