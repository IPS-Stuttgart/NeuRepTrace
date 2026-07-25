from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neureptrace.openneuro_real_shuffle_report import _nearest_row, write_real_shuffle_report


@pytest.mark.parametrize(
    "fixed_time",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        np.float64("nan"),
        np.asarray(np.inf),
    ],
)
def test_openneuro_nearest_row_rejects_nonfinite_requested_times(fixed_time: object) -> None:
    time_course = pd.DataFrame(
        {
            "time": [-0.1, 0.1],
            "balanced_accuracy": [0.4, 0.6],
        }
    )

    with pytest.raises(ValueError, match="fixed_time must be a finite real scalar"):
        _nearest_row(time_course, fixed_time)  # type: ignore[arg-type]


@pytest.mark.parametrize("fixed_time", [float("nan"), float("inf"), float("-inf")])
def test_openneuro_report_rejects_nonfinite_fixed_time_before_artifact_lookup(
    tmp_path: Path,
    fixed_time: float,
) -> None:
    out_dir = tmp_path / "report"

    with pytest.raises(ValueError, match="fixed_time must be a finite real scalar"):
        write_real_shuffle_report(
            real_dir=tmp_path / "missing-real",
            shuffle_dir=tmp_path / "missing-shuffle",
            out_dir=out_dir,
            fixed_time=fixed_time,
        )

    assert not out_dir.exists()


def test_openneuro_nearest_row_accepts_zero_dimensional_finite_time() -> None:
    time_course = pd.DataFrame(
        {
            "time": [-0.1, 0.1],
            "balanced_accuracy": [0.4, 0.6],
        }
    )

    row = _nearest_row(time_course, np.asarray(0.08))

    assert row["time"] == pytest.approx(0.1)
