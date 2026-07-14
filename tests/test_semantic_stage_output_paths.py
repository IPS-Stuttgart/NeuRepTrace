from __future__ import annotations

from pathlib import Path

import pytest

from neureptrace.semantic_stages import analyze_semantic_stages


@pytest.mark.parametrize(
    ("first_output", "second_output"),
    [
        ("out_time", "out_stages"),
        ("out_time", "out_report"),
        ("out_stages", "out_report"),
    ],
)
def test_analyze_semantic_stages_rejects_colliding_outputs_before_reading(
    tmp_path: Path,
    first_output: str,
    second_output: str,
) -> None:
    shared_output = tmp_path / "shared-output"
    normalized_alias = tmp_path / "nested" / ".." / "shared-output"
    outputs = {
        first_output: shared_output,
        second_output: normalized_alias,
    }

    with pytest.raises(ValueError, match="Semantic-stage output paths must be distinct"):
        analyze_semantic_stages(
            [tmp_path / "missing-state-traces.csv"],
            **outputs,
        )

    assert list(tmp_path.iterdir()) == []
