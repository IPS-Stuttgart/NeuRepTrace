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


def test_decode_from_config_null_dataset_name_uses_stable_output_token(tmp_path: Path) -> None:
    from neureptrace.decode_from_config import _decode_kwargs

    kwargs = _decode_kwargs(
        {
            "paths": {"base": "config_dir"},
            "dataset": {"name": None},
            "preprocessing": {},
            "decoding": {"label_column": "condition"},
            "outputs": {
                "base_dir": "results/{dataset}",
                "summary_csv": "{dataset}_summary.csv",
                "calibration_csv": "{dataset}_calibration.csv",
            },
        },
        config_dir=tmp_path,
    )

    assert kwargs["dataset_name"] == ""
    assert kwargs["out_path"] == tmp_path / "results/dataset/dataset_summary.csv"
    assert kwargs["calibration_out_path"] == tmp_path / "results/dataset/dataset_calibration.csv"


@pytest.mark.parametrize(
    ("preprocessing", "decoding", "message"),
    [
        ({"window_ms": True}, {}, "preprocessing.window_ms"),
        ({"window_size": float("inf")}, {}, "preprocessing.window_size"),
        ({"step_ms": 0}, {}, "preprocessing.step_ms"),
        ({"baseline_window": [False, 0.0]}, {}, "preprocessing.baseline_window"),
        ({}, {"decode_window": [0.0, float("nan")]}, "decoding.decode_window"),
        ({}, {"temporal_train_window": [True, 0.2]}, "decoding.temporal_train_window"),
        ({}, {"n_splits": True}, "decoding.n_splits"),
        ({}, {"max_iter": 100.5}, "decoding.max_iter"),
        ({}, {"tuning_cv_splits": 0}, "decoding.tuning_cv_splits"),
        ({}, {"calibration_bins": float("inf")}, "decoding.calibration_bins"),
        ({}, {"source_time_selection_output_time": True}, "decoding.source_time_selection_output_time"),
        ({}, {"alignment_repetition_cap": 0}, "decoding.alignment_repetition_cap"),
        ({}, {"alignment_components": 64.5}, "decoding.alignment_components"),
        ({}, {"alignment_target_calibration_seed": -1}, "decoding.alignment_target_calibration_seed"),
        ({}, {"label_shuffle_seed": True}, "decoding.label_shuffle_seed"),
        ({}, {"pseudo_label_confidence_threshold": True}, "pseudo_label_confidence_threshold"),
        ({}, {"pseudo_label_max_iterations": 0}, "decoding.pseudo_label_max_iterations"),
        ({}, {"pseudo_label_min_new": 1.5}, "decoding.pseudo_label_min_new"),
    ],
)
def test_decode_from_config_rejects_malformed_result_controls(
    tmp_path: Path,
    preprocessing: dict,
    decoding: dict,
    message: str,
) -> None:
    from neureptrace.decode_from_config import _decode_kwargs

    config = {
        "dataset": {"name": "synthetic"},
        "preprocessing": preprocessing,
        "decoding": {"label_column": "condition", **decoding},
        "outputs": {"base_dir": tmp_path.as_posix()},
    }

    with pytest.raises(ValueError, match=message):
        _decode_kwargs(config, config_dir=tmp_path)


@pytest.mark.parametrize(
    ("decoding", "message"),
    [
        ({"dann_max_epochs": True}, "decoding.dann_max_epochs"),
        ({"dann_batch_size": 12.5}, "decoding.dann_batch_size"),
        ({"dann_learning_rate": 0.0}, "decoding.dann_learning_rate"),
        ({"dann_weight_decay": -0.1}, "decoding.dann_weight_decay"),
        ({"dann_validation_fraction": 0.0}, "decoding.dann_validation_fraction"),
        ({"dann_dropout": 1.0}, "decoding.dann_dropout"),
        ({"dann_random_state": True}, "decoding.dann_random_state"),
    ],
)
def test_decode_from_config_rejects_malformed_dann_controls(
    tmp_path: Path,
    decoding: dict,
    message: str,
) -> None:
    from neureptrace.decode_from_config import _decode_kwargs

    config = {
        "dataset": {"name": "synthetic"},
        "preprocessing": {},
        "decoding": {"label_column": "condition", "decoder": "dann", **decoding},
        "outputs": {"base_dir": tmp_path.as_posix()},
    }

    with pytest.raises(ValueError, match=message):
        _decode_kwargs(config, config_dir=tmp_path)


def test_config_workflow_accepts_numeric_boolean_flags() -> None:
    from neureptrace.config_workflow import _as_bool

    assert _as_bool(1) is True
    assert _as_bool(0) is False


@pytest.mark.parametrize("value", [2, -1, 0.5])
def test_config_workflow_rejects_ambiguous_numeric_boolean_flags(value: float) -> None:
    from neureptrace.config_workflow import _as_bool

    with pytest.raises(DatasetConfigError, match="boolean"):
        _as_bool(value)
