from __future__ import annotations

from pathlib import Path

import matplotlib.axes
import numpy as np
import pandas as pd

from neureptrace.onset_sensitivity import plot_sensitivity_summary, summarize_sensitivity


def _sensitivity_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "task": ["motor", "motor"],
            "decoder": [np.nan, np.nan],
            "setting_id": ["setting-a", "setting-b"],
            "post_detection_latency_median": [0.10, 0.20],
            "false_alarm_rate": [0.0, 0.1],
            "post_zero_detected_rate": [0.8, 0.9],
            "correct_detection_rate": [0.7, 0.8],
        }
    )


def test_summarize_sensitivity_keeps_missing_optional_group_values() -> None:
    result = summarize_sensitivity(_sensitivity_rows())

    assert len(result) == 1
    assert result.loc[0, "task"] == "motor"
    assert pd.isna(result.loc[0, "decoder"])
    assert result.loc[0, "n_settings"] == 2
    assert np.isclose(result.loc[0, "false_alarm_rate_mean"], 0.05)


def test_plot_sensitivity_keeps_missing_optional_group_values(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plotted: list[tuple[object, ...]] = []
    original_plot = matplotlib.axes.Axes.plot

    def recording_plot(self, *args, **kwargs):
        plotted.append(args)
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", recording_plot)
    out_path = tmp_path / "sensitivity.png"

    result = plot_sensitivity_summary(_sensitivity_rows(), out_path)

    assert result == out_path
    assert out_path.is_file()
    assert len(plotted) == 2
