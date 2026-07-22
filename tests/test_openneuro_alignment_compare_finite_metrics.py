import numpy as np
import pandas as pd
import pytest

from neureptrace.openneuro_alignment_compare import _select_metric


@pytest.mark.parametrize(
    ("metric", "scores", "expected_value"),
    [
        ("balanced_accuracy", [0.60, np.inf, 0.99, 0.70], 0.70),
        ("brier", [0.30, -np.inf, 0.01, 0.20], 0.20),
    ],
)
def test_select_metric_ignores_non_finite_time_and_score_rows(
    metric: str,
    scores: list[float],
    expected_value: float,
) -> None:
    summary = pd.DataFrame(
        {
            "time": [0.1, 0.2, np.inf, 0.3],
            metric: scores,
        }
    )

    selected = _select_metric(summary, metric=metric, fixed_time=None)

    assert selected["selection_mode"] == "best_time"
    assert selected["selection_time"] == 0.3
    assert selected["selection_value"] == expected_value


def test_select_metric_reports_missing_when_no_finite_rows_remain() -> None:
    summary = pd.DataFrame(
        {
            "time": [np.inf, 0.2],
            "balanced_accuracy": [0.8, np.nan],
        }
    )

    selected = _select_metric(summary, metric="balanced_accuracy", fixed_time=None)

    assert selected["selection_mode"] == "missing_metric_rows"
    assert selected["selection_time"] == ""
    assert selected["selection_value"] == ""


def test_select_metric_rejects_non_finite_fixed_time() -> None:
    summary = pd.DataFrame(
        {
            "time": [0.1, 0.2],
            "balanced_accuracy": [0.6, 0.7],
        }
    )

    with pytest.raises(ValueError, match="fixed_time must be finite"):
        _select_metric(summary, metric="balanced_accuracy", fixed_time=np.nan)
