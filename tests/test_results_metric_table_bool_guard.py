from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.results import peak_metric_rows, subject_time_metrics, summarize_metric_table
from neureptrace.results.tables import summarize_metric_table as table_summarize_metric_table


def _time_decode_result_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": ["s1", "s1"],
            "fold": [0, 1],
            "time": [0.1, 0.1],
            "accuracy": [0.60, 0.70],
            "log_loss": [0.5, 0.4],
            "brier": [0.3, 0.2],
            "ece": [0.1, 0.2],
        }
    )


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


@pytest.mark.parametrize("summarizer", [summarize_metric_table, table_summarize_metric_table])
@pytest.mark.parametrize(
    ("column", "kwargs"),
    [
        ("accuracy", {}),
        ("chance", {"chance_column": "chance"}),
        ("permutation_p_value", {"permutation_p_column": "permutation_p_value"}),
        ("n_validation_classes", {"chance_column": "chance", "chance_class_columns": ("n_validation_classes",)}),
    ],
)
def test_summarize_metric_table_rejects_array_valued_numeric_cells(summarizer, column, kwargs) -> None:
    frame = pd.DataFrame(
        {
            "decoder": ["logistic"],
            "accuracy": [0.60],
            "chance": [0.50],
            "permutation_p_value": [0.04],
            "n_validation_classes": [2],
        }
    )
    frame[column] = pd.Series([np.asarray(frame[column].iloc[0])], dtype=object)

    with pytest.raises(ValueError, match=f"Metric table column '{column}' must not contain arrays"):
        summarizer(frame, "accuracy", "decoder", **kwargs)


@pytest.mark.parametrize("ece_bins", [np.asarray(10), np.array([10]), np.asarray(True), np.array([True])])
def test_subject_time_metrics_rejects_array_valued_ece_bins(ece_bins) -> None:
    with pytest.raises(ValueError, match="ece_bins must be a positive integer"):
        subject_time_metrics(_time_decode_result_frame(), ece_bins=ece_bins)


@pytest.mark.parametrize("prefer_time", [np.asarray(0.0), np.array([0.0]), np.asarray(True), np.array([True])])
def test_peak_metric_rows_rejects_array_valued_prefer_time(prefer_time) -> None:
    frame = pd.DataFrame(
        {
            "decoder": ["logistic", "logistic"],
            "time": [0.1, 0.2],
            "accuracy": [0.60, 0.70],
        }
    )

    with pytest.raises(ValueError, match="prefer_time must be a finite numeric value"):
        peak_metric_rows(frame, "accuracy", ("decoder",), prefer_time=prefer_time)
