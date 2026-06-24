import numpy as np
import pandas as pd

from neureptrace.results import summarize_metric_table


def test_summarize_metric_table_treats_boolean_chance_fields_as_missing():
    frame = pd.DataFrame(
        {
            "decoder": ["bools", "bools", "valid"],
            "accuracy": [0.75, 0.25, 0.60],
            "chance_accuracy": pd.Series([True, np.bool_(False), None], dtype=object),
            "n_validation_classes": pd.Series([np.bool_(True), False, 4], dtype=object),
        }
    )

    summary = summarize_metric_table(
        frame,
        "accuracy",
        "decoder",
        chance_column="chance_accuracy",
        chance_class_columns=("n_validation_classes",),
    )

    bools = summary.loc[summary["decoder"] == "bools"].iloc[0]
    assert pd.isna(bools["chance_accuracy_mean"])
    assert pd.isna(bools["chance_accuracy_min"])
    assert pd.isna(bools["chance_accuracy_max"])
    assert pd.isna(bools["chance_classes_mean"])
    assert bools["chance_classes_counts"] == ""
    assert bools["accuracy_above_chance_count"] == 0

    valid = summary.loc[summary["decoder"] == "valid"].iloc[0]
    assert valid["chance_accuracy_mean"] == 0.25
    assert valid["chance_accuracy_min"] == 0.25
    assert valid["chance_accuracy_max"] == 0.25
    assert valid["chance_classes_mean"] == 4.0
    assert valid["chance_classes_counts"] == "4:1"
    assert valid["accuracy_above_chance_count"] == 1
