from __future__ import annotations

import pandas as pd
import pytest

from neureptrace.openneuro_real_shuffle_report import _best_time_row, _nearest_row


def test_openneuro_real_shuffle_time_selection_ignores_nonfinite_rows() -> None:
    frame = pd.DataFrame(
        {
            "time": [float("nan"), 0.184, 0.232],
            "balanced_accuracy": [0.99, 0.50, 0.60],
        }
    )

    nearest = _nearest_row(frame, 0.18)
    best = _best_time_row(frame)

    assert nearest["time"] == pytest.approx(0.184)
    assert best["time"] == pytest.approx(0.232)
    assert best["balanced_accuracy"] == pytest.approx(0.60)


def test_openneuro_real_shuffle_time_selection_rejects_all_nonfinite_rows() -> None:
    frame = pd.DataFrame(
        {
            "time": [float("nan"), float("inf")],
            "balanced_accuracy": [0.40, 0.50],
        }
    )

    with pytest.raises(ValueError, match="finite time"):
        _nearest_row(frame, 0.184)
    with pytest.raises(ValueError, match="finite 'balanced_accuracy'"):
        _best_time_row(frame)
