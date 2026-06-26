from __future__ import annotations

import json
from pathlib import Path

import pytest

from neureptrace.config_workflow import validate_dataset_config


def _write_workflow_config(tmp_path: Path, *, preprocessing: dict) -> Path:
    config_path = tmp_path / "workflow.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset": {"epochs": "subject-01-epo.fif"},
                "decoding": {"label_column": "stimulus"},
                "preprocessing": preprocessing,
                "outputs": {"metrics_csv": "results/metrics.csv"},
            }
        ),
        encoding="utf-8",
    )
    return config_path


@pytest.mark.parametrize(
    "window_value",
    [
        [False, 0.0],
        [float("nan"), 0.0],
        ["bad", 0.0],
    ],
)
def test_validate_dataset_config_rejects_malformed_float_windows(tmp_path: Path, window_value: list[object]) -> None:
    config_path = _write_workflow_config(tmp_path, preprocessing={"baseline_window": window_value})

    problems = validate_dataset_config(config_path, check_files=False)

    assert problems
    assert "baseline_window" in problems[0]
    assert "finite numeric values" in problems[0]
