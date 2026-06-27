from __future__ import annotations

import pandas as pd
import pytest

from neureptrace.loso_observation_diagnostics import _best_time


def test_loso_best_time_ignores_nonfinite_time_and_metric_rows() -> None:
    summary = pd.DataFrame(
        {
            "time": [float("nan"), 0.10, 0.20, float("inf")],
            "balanced_accuracy": [0.99, float("nan"), 0.55, 0.88],
        }
    )

    assert _best_time(summary, "balanced_accuracy") == pytest.approx(0.20)


def test_loso_best_time_filters_nonfinite_candidates_for_minimized_metric() -> None:
    summary = pd.DataFrame(
        {
            "time": [float("nan"), 0.10, 0.20],
            "log_loss": [0.0, float("inf"), 1.2],
        }
    )

    assert _best_time(summary, "log_loss") == pytest.approx(0.20)


def test_loso_best_time_rejects_all_nonfinite_candidates() -> None:
    summary = pd.DataFrame(
        {
            "time": [float("nan"), 0.10, float("inf")],
            "balanced_accuracy": [0.90, float("nan"), 0.80],
        }
    )

    with pytest.raises(ValueError, match="finite 'balanced_accuracy'"):
        _best_time(summary, "balanced_accuracy")
