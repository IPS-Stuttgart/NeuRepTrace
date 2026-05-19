from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from neureptrace import dataset_workflow as dw


def _write_dataset_config(path: Path, text: str) -> Path:
    path.write_text(dedent(text).strip() + "\n", encoding="utf-8")
    return path


def test_dataset_config_expands_participants_and_paths(tmp_path: Path):
    config_path = _write_dataset_config(
        tmp_path / "dataset.yml",
        """
        schema_version: neureptrace.dataset.v1
        dataset:
          id: bush_meg
          root: data
        loader:
          input_format: mne-epochs
          label_column: condition
        participants:
          include: [10, "13-14"]
        recordings:
          main:
            pattern: "Part{participant}-epo.fif"
            metadata_csv: "Part{participant}_metadata.csv"
        analyses:
          - name: stimulus_cv
            task: time_decode
            recording: main
            output_csv: "outputs/{analysis}_{subject}.csv"
            observations_out: true
            window_ms: 100
            step_ms: 50
            decoder: linear_svm
            emission_mode: calibrated
        """,
    )

    config = dw.load_dataset_config(config_path)
    plans = dw.plan_dataset_runs(config)

    assert [plan.participant for plan in plans] == [10, 13, 14]
    assert plans[0].subject == "Part10"
    assert plans[0].recording_path == tmp_path / "data" / "Part10-epo.fif"
    assert plans[0].metadata_csv == tmp_path / "data" / "Part10_metadata.csv"
    assert plans[0].output_csv == tmp_path / "outputs" / "stimulus_cv_Part10.csv"
    assert plans[0].observations_out == tmp_path / "outputs" / "stimulus_cv_Part10_observations.csv"
    assert plans[0].options["window_ms"] == 100
    assert plans[0].options["step_ms"] == 50


def test_run_dataset_config_calls_existing_time_decoder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_path = _write_dataset_config(
        tmp_path / "dataset.yml",
        """
        schema_version: neureptrace.dataset.v1
        dataset:
          id: bush_meg
          root: data
        loader:
          input_format: mne-epochs
        participants: [10]
        recordings:
          main:
            pattern: "Part{participant}-epo.fif"
            metadata_csv: "Part{participant}_metadata.csv"
        analyses:
          - name: stimulus_cv
            recording: main
            label_column: condition
            output_csv: "outputs/{participant}_{analysis}.csv"
            observations_out: "outputs/{participant}_{analysis}_observations.csv"
            group_column: session
            picks: data
            tmin: -0.1
            tmax: 0.5
            window_ms: 100
            step_ms: 50
            n_splits: 3
            decoder: linear_svm
            emission_mode: calibrated
            feature_preprocessor: pca_whiten
            pca_components: 0.99
            normalization: subject_baseline_whiten
            baseline_window: [-0.35, -0.05]
        """,
    )
    calls: list[dict] = []

    def fake_decode(**kwargs):
        calls.append(kwargs)
        return None

    monkeypatch.setattr(dw, "_run_time_resolved_decode", fake_decode)

    results = dw.run_dataset_config(config_path)

    assert len(results) == 1
    assert results[0].executed is True
    call = calls[0]
    assert call["epochs_path"] == tmp_path / "data" / "Part10-epo.fif"
    assert call["metadata_csv"] == tmp_path / "data" / "Part10_metadata.csv"
    assert call["label_column"] == "condition"
    assert call["out_path"] == tmp_path / "outputs" / "10_stimulus_cv.csv"
    assert call["observation_out_path"] == tmp_path / "outputs" / "10_stimulus_cv_observations.csv"
    assert call["subject"] == "Part10"
    assert call["group_column"] == "session"
    assert call["decoder"] == "linear_svm"
    assert call["feature_preprocessor"] == "pca_whiten"
    assert call["pca_components"] == 0.99
    assert call["normalization"] == "subject_baseline_whiten"


def test_fieldtrip_mat_runs_are_staged_before_decoding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_path = _write_dataset_config(
        tmp_path / "dataset.yml",
        """
        schema_version: neureptrace.dataset.v1
        dataset:
          id: bush_meg
          root: data
        loader:
          format: mat
          structure: fieldtrip_raw
          root_path: [data, 0]
          label_base: 1
          trim_overlong_labels: true
          ch_type: grad
        participants: [10]
        recordings:
          main:
            pattern: "Part{participant}Data.mat"
        analyses:
          - name: stimulus_cv
            recording: main
            output_csv: "outputs/{analysis}_{subject}.csv"
        """,
    )
    staged: list[dict] = []
    calls: list[dict] = []

    def fake_stage(**kwargs):
        staged.append(kwargs)
        return tmp_path / "staged-epo.fif", tmp_path / "staged_metadata.csv"

    def fake_decode(**kwargs):
        calls.append(kwargs)
        return None

    monkeypatch.setattr(dw, "_stage_fieldtrip_mat", fake_stage)
    monkeypatch.setattr(dw, "_run_time_resolved_decode", fake_decode)

    dw.run_dataset_config(config_path, staging_dir=tmp_path / "stage")

    assert staged[0]["plan"].input_format == "fieldtrip-mat"
    assert staged[0]["plan"].recording_path == tmp_path / "data" / "Part10Data.mat"
    assert staged[0]["config"].loader.root_path == ("data", 0)
    assert calls[0]["epochs_path"] == tmp_path / "staged-epo.fif"
    assert calls[0]["metadata_csv"] == tmp_path / "staged_metadata.csv"


def test_duplicate_output_paths_are_rejected(tmp_path: Path):
    config_path = _write_dataset_config(
        tmp_path / "dataset.yml",
        """
        dataset:
          root: data
        participants: [10, 11]
        recordings:
          main:
            pattern: "Part{participant}-epo.fif"
        analyses:
          - name: stimulus_cv
            recording: main
            output_csv: "outputs/stimulus.csv"
        """,
    )

    with pytest.raises(ValueError, match="same output CSV"):
        dw.plan_dataset_runs(config_path)
