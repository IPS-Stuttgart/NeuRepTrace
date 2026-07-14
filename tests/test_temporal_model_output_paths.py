from __future__ import annotations

from pathlib import Path

import pytest

from neureptrace.temporal_model import fit_temporal_models


def test_fit_temporal_models_rejects_colliding_output_paths_before_reading(tmp_path: Path) -> None:
    shared_output = tmp_path / "shared.csv"
    normalized_alias = tmp_path / "nested" / ".." / "shared.csv"

    with pytest.raises(ValueError, match="summary and state output paths must be distinct"):
        fit_temporal_models(
            [tmp_path / "missing-observations.csv"],
            out_summary=shared_output,
            out_states=normalized_alias,
        )

    assert list(tmp_path.iterdir()) == []
