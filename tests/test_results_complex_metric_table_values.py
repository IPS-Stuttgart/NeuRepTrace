from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.results import summarize_metric_table


def test_summarize_metric_table_rejects_complex_metric_values() -> None:
    frame = pd.DataFrame(
        {
            "accuracy": pd.Series(
                [0.75, np.complex128(0.25 + 0.5j)],
                dtype=object,
            )
        }
    )

    with pytest.raises(ValueError, match=r"Metric table column 'accuracy'.*complex"):
        summarize_metric_table(frame, "accuracy", None)


@pytest.mark.parametrize(
    ("column", "kwargs"),
    [
        ("chance_accuracy", {"chance_column": "chance_accuracy"}),
        ("permutation_p", {"permutation_p_column": "permutation_p"}),
        (
            "n_validation_classes",
            {
                "chance_column": "chance_accuracy",
                "chance_class_columns": ("n_validation_classes",),
            },
        ),
    ],
)
def test_summarize_metric_table_rejects_complex_optional_numeric_values(
    column: str,
    kwargs: dict[str, object],
) -> None:
    frame = pd.DataFrame(
        {
            "accuracy": [0.75, 0.25],
            "chance_accuracy": pd.Series([0.5, 0.5], dtype=object),
            "permutation_p": pd.Series([0.1, 0.2], dtype=object),
            "n_validation_classes": pd.Series([2, 2], dtype=object),
        }
    )
    frame.at[1, column] = np.complex64(1.0 + 0.25j)

    with pytest.raises(ValueError, match=rf"Metric table column '{column}'.*complex"):
        summarize_metric_table(frame, "accuracy", None, **kwargs)
