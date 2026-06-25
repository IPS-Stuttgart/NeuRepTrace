from __future__ import annotations

import json

import pytest

from neureptrace.config_workflow import DatasetConfigError, _as_bool, run_decode_from_config


@pytest.mark.parametrize(("value", "expected"), [(1.0, True), (0.0, False)])
def test_config_workflow_accepts_float_boolean_flags(value: float, expected: bool) -> None:
    assert _as_bool(value) is expected


@pytest.mark.parametrize("value", [0.5, 2.0, -1.0, float("nan"), float("inf")])
def test_config_workflow_rejects_ambiguous_float_boolean_flags(value: float) -> None:
    with pytest.raises(DatasetConfigError, match="boolean"):
        _as_bool(value)


def test_run_decode_from_config_accepts_float_boolean_flags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
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
                    "label_shuffle_control": 0.0,
                },
                "outputs": {"metrics_csv": "results/metrics.csv"},
                "tuning": {"enabled": 1.0},
            }
        ),
        encoding="utf-8",
    )

    run_decode_from_config(config_path)

    assert captured["tune_hyperparameters"] is True
    assert captured["label_shuffle_control"] is False
