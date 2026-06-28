from __future__ import annotations

import json
from pathlib import Path

import pytest

from neureptrace.config_workflow import DatasetConfigError, _as_float_pair, run_decode_from_config, validate_dataset_config


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("-0.2,0.0", (-0.2, 0.0)),
        ("[-0.2, 0.0]", (-0.2, 0.0)),
        ("-0.2 0.0", (-0.2, 0.0)),
        ("", None),
    ],
)
def test_config_workflow_float_pair_accepts_string_window_forms(value: str, expected: tuple[float, float] | None) -> None:
    assert _as_float_pair(value, name="baseline_window") == expected


@pytest.mark.parametrize("value", ["0.0,0.1,0.2", "[0.0, bad]"])
def test_config_workflow_float_pair_rejects_malformed_strings(value: str) -> None:
    with pytest.raises(DatasetConfigError, match="baseline_window"):
        _as_float_pair(value, name="baseline_window")


def test_legacy_config_workflow_accepts_string_window_controls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_run_time_resolved_decode(**kwargs):  # noqa: ANN202, ANN003
        captured.update(kwargs)
        return []

    monkeypatch.setattr("neureptrace.config_workflow.run_time_resolved_decode", fake_run_time_resolved_decode)
    config_path = tmp_path / "workflow.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset": {"name": "synthetic", "epochs": "subject-01-epo.fif"},
                "decoding": {
                    "label_column": "stimulus",
                    "temporal_train_window": "0.1 0.2",
                },
                "preprocessing": {"baseline_window": "[-0.2, 0.0]"},
                "outputs": {"metrics_csv": "results/metrics.csv"},
            }
        ),
        encoding="utf-8",
    )

    assert validate_dataset_config(config_path, check_files=False) == []

    run_decode_from_config(config_path)

    assert captured["baseline_window"] == (-0.2, 0.0)
    assert captured["temporal_train_window"] == (0.1, 0.2)
