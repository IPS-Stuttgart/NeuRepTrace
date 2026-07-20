from __future__ import annotations

import pandas as pd
import pytest

from neureptrace.results import mean_across_folds


@pytest.mark.parametrize("include_weights", [False, True])
def test_mean_across_folds_rejects_duplicate_metric_columns(include_weights: bool) -> None:
    frame = pd.DataFrame(
        {
            "subject": ["subject-1", "subject-1"],
            "time": [0.0, 0.0],
            "fold": [1, 2],
            "accuracy": [0.5, 0.7],
        }
    )
    if include_weights:
        frame["n_test"] = [10, 20]

    with pytest.raises(ValueError, match="metric_columns must not contain duplicates.*accuracy"):
        mean_across_folds(
            frame,
            ["subject", "time"],
            metric_columns=["accuracy", "accuracy"],
        )
