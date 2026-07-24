from __future__ import annotations

import json
from pathlib import Path

import pytest

from neureptrace import config_workflow


def _write_config(tmp_path: Path, outputs: dict[str, str]) -> Path:
    config_path = tmp_path / "workflow.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset": {"epochs": "subject-01-epo.fif"},
                "decoding": {"label_column": "stimulus"},
                "outputs": outputs,
            }
        ),
        encoding="utf-8",
    )
    return config_path


@pytest.mark.parametrize(
    ("colliding_output", "path"),
    [
        ("calibration_csv", "results/metrics.csv"),
        ("observations_csv", "results/intermediate/../metrics.csv"),
    ],
)
def test_validate_dataset_config_rejects_colliding_output_paths(
    tmp_path: Path,
    colliding_output: str,
    path: str,
) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "metrics_csv": "results/metrics.csv",
            colliding_output: path,
        },
    )

    problems = config_workflow.validate_dataset_config(config_path, check_files=False)

    assert len(problems) == 1
    assert "Configured output paths must be distinct" in problems[0]
    assert "outputs.metrics_csv" in problems[0]
    assert f"outputs.{colliding_output}" in problems[0]


def test_run_decode_from_config_rejects_collision_before_decoder_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "metrics_csv": "results/metrics.csv",
            "calibration_csv": "results/metrics.csv",
        },
    )

    def unexpected_decode(**_kwargs):
        raise AssertionError("decoder must not run for colliding output paths")

    monkeypatch.setattr(config_workflow, "run_time_resolved_decode", unexpected_decode)

    with pytest.raises(config_workflow.DatasetConfigError, match="output paths must be distinct"):
        config_workflow.run_decode_from_config(config_path)
