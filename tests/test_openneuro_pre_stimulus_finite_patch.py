from __future__ import annotations

import math

import pandas as pd
import pytest

from neureptrace.openneuro_real_shuffle_report import _pre_stimulus_summary


def test_openneuro_pre_stimulus_summary_ignores_nonfinite_metric_rows() -> None:
    frame = pd.DataFrame(
        {
            "time": [-0.2, -0.1, float("nan"), 0.1],
            "balanced_accuracy": [float("inf"), 0.4, 0.99, 0.7],
            "top2_accuracy": [0.9, float("nan"), 0.99, 0.8],
        }
    )

    summary = _pre_stimulus_summary(frame)

    assert summary["n_pre_stimulus_times"] == 2
    assert summary["pre_stimulus_balanced_accuracy_mean"] == pytest.approx(0.4)
    assert summary["pre_stimulus_balanced_accuracy_max"] == pytest.approx(0.4)
    assert summary["pre_stimulus_balanced_accuracy_max_time"] == pytest.approx(-0.1)
    assert summary["pre_stimulus_top2_accuracy_mean"] == pytest.approx(0.9)
    assert summary["pre_stimulus_top2_accuracy_max"] == pytest.approx(0.9)


def test_openneuro_pre_stimulus_summary_handles_all_nonfinite_metrics() -> None:
    frame = pd.DataFrame(
        {
            "time": [-0.2, -0.1],
            "balanced_accuracy": [float("nan"), float("inf")],
            "top2_accuracy": [float("nan"), float("-inf")],
        }
    )

    summary = _pre_stimulus_summary(frame)

    assert summary["n_pre_stimulus_times"] == 2
    assert math.isnan(summary["pre_stimulus_balanced_accuracy_mean"])
    assert math.isnan(summary["pre_stimulus_balanced_accuracy_max"])
    assert math.isnan(summary["pre_stimulus_balanced_accuracy_max_time"])
    assert math.isnan(summary["pre_stimulus_top2_accuracy_mean"])
    assert math.isnan(summary["pre_stimulus_top2_accuracy_max"])
