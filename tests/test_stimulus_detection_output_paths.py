from __future__ import annotations

from pathlib import Path

import pytest

import neureptrace._stimulus_detection_public as stimulus_public


@pytest.mark.parametrize(
    ("first_name", "second_name"),
    [
        ("out_events", "out_summary"),
        ("out_events", "out_thresholds"),
        ("out_summary", "out_thresholds"),
    ],
)
def test_csv_stimulus_detection_rejects_colliding_output_paths(
    tmp_path: Path,
    first_name: str,
    second_name: str,
) -> None:
    shared = tmp_path / "shared.csv"
    kwargs = {
        "out_events": tmp_path / "events.csv",
        "out_summary": tmp_path / "summary.csv",
        "out_thresholds": tmp_path / "thresholds.csv",
    }
    kwargs[first_name] = shared
    kwargs[second_name] = shared.parent / "." / shared.name

    with pytest.raises(
        ValueError,
        match=rf"{first_name} and {second_name} both resolve",
    ):
        stimulus_public.detect_stimulus_events_from_csvs(
            [tmp_path / "missing-observations.csv"],
            **kwargs,
        )

    assert not shared.exists()
