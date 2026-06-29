from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.results import summarize_metric_table
from neureptrace.results.tables import summarize_metric_table as table_summarize_metric_table


@pytest.mark.parametrize("summarizer", [summarize_metric_table, table_summarize_metric_table])
def test_summarize_metric_table_rejects_boolean_metric_values(summarizer) -> None:
    frame = pd.DataFrame(
        {
            "decoder": ["logistic", "logistic"],
            "accuracy": pd.Series([np.bool_(True), 0.70], dtype=object),
            "chance": [0.50, 0.50],
        }
    )

    with pytest.raises(ValueError, match="Metric table column 'accuracy' must not contain booleans"):
        summarizer(frame, "accuracy", "decoder", chance_column="chance")


@pytest.mark.parametrize("summarizer", [summarize_metric_table, table_summarize_metric_table])
def test_summarize_metric_table_rejects_boolean_chance_values(summarizer) -> None:
    frame = pd.DataFrame(
        {
            "decoder": ["logistic", "logistic"],
            "accuracy": [0.60, 0.70],
            "chance": pd.Series([False, 0.50], dtype=object),
        }
    )

    with pytest.raises(ValueError, match="Metric table column 'chance' must not contain booleans"):
        summarizer(frame, "accuracy", "decoder", chance_column="chance")


@pytest.mark.parametrize("summarizer", [summarize_metric_table, table_summarize_metric_table])
def test_summarize_metric_table_rejects_boolean_permutation_p_values(summarizer) -> None:
    frame = pd.DataFrame(
        {
            "decoder": ["logistic", "logistic"],
            "accuracy": [0.60, 0.70],
            "permutation_p_value": pd.Series([False, 0.20], dtype=object),
        }
    )

    with pytest.raises(ValueError, match="Metric table column 'permutation_p_value' must not contain booleans"):
        summarizer(frame, "accuracy", "decoder", permutation_p_column="permutation_p_value")


@pytest.mark.parametrize("summarizer", [summarize_metric_table, table_summarize_metric_table])
def test_summarize_metric_table_rejects_boolean_chance_class_counts(summarizer) -> None:
    frame = pd.DataFrame(
        {
            "decoder": ["logistic", "logistic"],
            "accuracy": [0.60, 0.70],
            "chance_accuracy": [None, None],
            "n_validation_classes": pd.Series([True, 4], dtype=object),
        }
    )

    with pytest.raises(ValueError, match="Metric table column 'n_validation_classes' must not contain booleans"):
        summarizer(
            frame,
            "accuracy",
            "decoder",
            chance_column="chance_accuracy",
            chance_class_columns=("n_validation_classes",),
        )
