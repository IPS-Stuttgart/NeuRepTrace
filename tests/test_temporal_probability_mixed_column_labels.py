from __future__ import annotations

import pandas as pd
import pytest

from neureptrace.temporal_model import probability_columns
from neureptrace.temporal_smoothing import metrics_from_probability_observations


def test_temporal_probability_columns_ignore_non_string_metadata_labels() -> None:
    observations = pd.DataFrame(
        [["metadata", 7, 0.2, 0.8]],
        columns=pd.Index(
            [0, ("metadata", "run"), "prob_class_0", "prob_class_1"],
            dtype=object,
        ),
    )

    assert probability_columns(observations) == ["prob_class_0", "prob_class_1"]


def test_temporal_metrics_accept_mixed_column_label_types() -> None:
    observations = pd.DataFrame(
        {
            0: ["ignored", "ignored"],
            ("metadata", "run"): [7, 7],
            "subject": ["sub-01", "sub-01"],
            "fold": [0, 0],
            "time": [0.1, 0.1],
            "true_label": [0, 1],
            "prob_class_0": [0.9, 0.1],
            "prob_class_1": [0.1, 0.9],
        }
    )

    metrics = metrics_from_probability_observations(observations)

    assert metrics.loc[0, "accuracy"] == pytest.approx(1.0)
    assert metrics.loc[0, "n_test"] == 2


def test_temporal_probability_columns_keep_documented_missing_schema_error() -> None:
    observations = pd.DataFrame(
        [["metadata", 7]],
        columns=pd.Index([0, ("metadata", "run")], dtype=object),
    )

    with pytest.raises(ValueError, match="probability columns"):
        probability_columns(observations)
