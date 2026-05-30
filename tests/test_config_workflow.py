from __future__ import annotations

import json
from pathlib import Path

import pytest

from neureptrace.config_workflow import (
    DatasetConfigError,
    load_dataset_config,
    run_decode_from_config,
    validate_dataset_config,
)


def test_load_dataset_config_rejects_unknown_extension(tmp_path: Path) -> None:
    config_path = tmp_path / "workflow.toml"
    config_path.write_text("{}", encoding="utf-8")

    with pytest.raises(DatasetConfigError):
        load_dataset_config(config_path)


def test_validate_dataset_config_reports_missing_required_values(tmp_path: Path) -> None:
    config_path = tmp_path / "workflow.json"
    config_path.write_text(json.dumps({"dataset": {}}), encoding="utf-8")

    problems = validate_dataset_config(config_path, check_files=False)

    assert problems
    assert "dataset.epochs" in problems[0]


def test_validate_dataset_config_structure_without_files(tmp_path: Path) -> None:
    config_path = tmp_path / "workflow.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset": {
                    "epochs": "subject-01-epo.fif",
                    "metadata_csv": "metadata.csv",
                    "subject": "subject-01",
                },
                "decoding": {
                    "label_column": "stimulus",
                    "group_column": "run",
                    "decoder": "logistic",
                    "emission_mode": "calibrated",
                },
                "preprocessing": {
                    "window_ms": 20.0,
                    "step_ms": 10.0,
                    "baseline_window": [-0.35, -0.05],
                },
                "outputs": {
                    "metrics_csv": "results/metrics.csv",
                    "observations_csv": "results/observations.csv",
                },
            }
        ),
        encoding="utf-8",
    )

    assert validate_dataset_config(config_path, check_files=False) == []


def test_run_decode_from_config_translates_sections(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
                "dataset": {"epochs": "subject-01-epo.fif", "metadata_csv": "metadata.csv"},
                "decoding": {"label_column": "stimulus", "group_column": "run", "n_splits": 3},
                "preprocessing": {"normalization": "subject_baseline_z", "baseline_window": [-0.2, 0.0]},
                "outputs": {"metrics_csv": "results/metrics.csv", "calibration_csv": "results/calibration.csv"},
                "tuning": {"enabled": True, "cv_splits": 2, "c_grid": [0.1, 1.0]},
            }
        ),
        encoding="utf-8",
    )

    run_decode_from_config(config_path)

    assert captured["epochs_path"] == tmp_path / "subject-01-epo.fif"
    assert captured["metadata_csv"] == tmp_path / "metadata.csv"
    assert captured["out_path"] == tmp_path / "results/metrics.csv"
    assert captured["label_column"] == "stimulus"
    assert captured["group_column"] == "run"
    assert captured["baseline_window"] == (-0.2, 0.0)
    assert captured["tune_hyperparameters"] is True


def test_decode_from_config_quoted_false_flags_stay_disabled(tmp_path: Path) -> None:
    from neureptrace.decode_from_config import _decode_kwargs

    kwargs = _decode_kwargs(
        {
            "dataset": {"name": "synthetic"},
            "preprocessing": {},
            "decoding": {
                "label_column": "condition",
                "tune_hyperparameters": "false",
                "label_shuffle_control": "false",
            },
            "outputs": {"base_dir": tmp_path.as_posix()},
        },
        config_dir=tmp_path,
    )

    assert kwargs["tune_hyperparameters"] is False
    assert kwargs["label_shuffle_control"] is False
