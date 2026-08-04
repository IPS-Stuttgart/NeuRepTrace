from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.results import summarize_metric_table


def _metric_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decoder": ["a", "a", "b", "b"],
            "accuracy": [0.8, 0.7, 0.6, 0.5],
            "permutation_p": [0.01, 0.2, 0.04, 0.8],
        }
    )


def test_summarize_metric_table_reuses_one_pass_p_value_thresholds_for_every_group() -> None:
    thresholds = (threshold for threshold in (0.05, 0.01))

    summary = summarize_metric_table(
        _metric_frame(),
        "accuracy",
        "decoder",
        permutation_p_column="permutation_p",
        p_value_thresholds=thresholds,
    )

    assert summary["n_significant_p_0.05"].tolist() == [1, 1]
    assert summary["n_significant_p_0.01"].tolist() == [0, 0]


def test_summarize_metric_table_requires_requested_permutation_column() -> None:
    with pytest.raises(ValueError, match="missing_permutation_p"):
        summarize_metric_table(
            _metric_frame(),
            "accuracy",
            "decoder",
            permutation_p_column="missing_permutation_p",
        )


@pytest.mark.parametrize(
    "thresholds",
    ([0.0], [1.01], [True], [np.complex128(0.05 + 0.01j)], "0.05"),
)
def test_summarize_metric_table_rejects_invalid_p_value_thresholds(thresholds: object) -> None:
    with pytest.raises(ValueError, match="p_value_thresholds"):
        summarize_metric_table(
            _metric_frame(),
            "accuracy",
            "decoder",
            permutation_p_column="permutation_p",
            p_value_thresholds=thresholds,
        )
