from __future__ import annotations

import json
from pathlib import Path

import pytest

from neureptrace.config_workflow import DatasetConfigError, _as_bool, run_decode_from_config


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, True),
        (0, False),
        (1.0, True),
        (0.0, False),
    ],
)
def test_config_workflow_as_bool_accepts_numeric_flags(value, expected) -> None:
    assert _as_bool(value) is expected


@pytest.mark.parametrize("value", [2, -1, 0.5, float("nan")])
def test_config_workflow_as_bool_rejects_ambiguous_numeric_flags(value) -> None:
    with pytest.raises(DatasetConfigError, match="boolean"):
        _as_bool(value)


def test_config_workflow_accepts_numeric_boolean_flags_in_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured = {}

    def fake_run_time_resolved_decode(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        "neureptrace.config_workflow.run_time_resolved_decode",
        fake_run_time_resolved_decode,
    )

    config_path = tmp_path / "workflow.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset": {"epochs": "subject-01-epo.fif"},
                "decoding": {
                    "label_column": "stimulus",
                    "label_shuffle_control": 0,
                },
                "outputs": {"metrics_csv": "results/metrics.csv"},
                "tuning": {"enabled": 1},
            }
        ),
        encoding="utf-8",
    )

    run_decode_from_config(config_path)

    assert captured["tune_hyperparameters"] is True
    assert captured["label_shuffle_control"] is False
