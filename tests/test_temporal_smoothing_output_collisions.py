from __future__ import annotations

from pathlib import Path

import pytest

from neureptrace import temporal_smoothing


def test_temporal_smoothing_rejects_aliased_outputs_before_reading_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "results" / "smoothed.csv"
    aliased_metrics_path = tmp_path / "results" / "nested" / ".." / "smoothed.csv"

    def unexpected_read(_paths: list[Path]):
        raise AssertionError("input observations must not be read for colliding outputs")

    monkeypatch.setattr(temporal_smoothing, "read_probability_observations", unexpected_read)

    with pytest.raises(ValueError, match="observation and metric output paths must be distinct"):
        temporal_smoothing.smooth_probability_observations(
            [tmp_path / "missing-observations.csv"],
            out_observations=output_path,
            out_metrics=aliased_metrics_path,
        )

    assert not output_path.exists()
    assert not output_path.parent.exists()
