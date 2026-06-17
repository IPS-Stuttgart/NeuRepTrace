from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neureptrace.loso_time_decode import run_loso_time_decode


@dataclass
class FakeEpochDataset:
    data: np.ndarray
    times: np.ndarray
    metadata: pd.DataFrame
    name: str = "fake_loso"


def _fake_bush_like_dataset() -> FakeEpochDataset:
    rng = np.random.default_rng(42)
    subjects = np.repeat(["s1", "s2", "s3", "s4"], 4)
    labels = np.tile(["class_a", "class_b", "class_a", "class_b"], 4)
    data = rng.normal(scale=0.05, size=(len(labels), 2, 5))
    sign = np.where(labels == "class_a", -1.0, 1.0)
    data[:, 0, :2] += 2.0 * sign[:, None]
    data[:, 1, 2:] += 0.15 * sign[:, None]
    metadata = pd.DataFrame({"subject": subjects, "condition": labels, "split": "main"})
    return FakeEpochDataset(data=data, times=np.array([0.00, 0.01, 0.02, 0.03, 0.04]), metadata=metadata)


def _loso_config(tmp_path: Path, *, preprocessing: dict | None = None, loso: dict | None = None) -> dict:
    return {
        "dataset": {"name": "fake_loso"},
        "preprocessing": {"window_ms": 20.0, "step_ms": 20.0, "normalization": "none", **(preprocessing or {})},
        "loso": {
            "label_column": "condition",
            "group_column": "subject",
            "trial_filter": {"split": "main"},
            "decoder": "logistic",
            "emission_mode": "uncalibrated",
            "normalization_scope": "per_group",
            "max_iter": 500,
            **(loso or {}),
        },
        "outputs": {"summary_csv": str(tmp_path / "loso.csv"), "provenance": False},
    }


def test_loso_time_decode_runs_source_only_subject_folds(tmp_path: Path, monkeypatch):
    summary = tmp_path / "loso.csv"
    observations = tmp_path / "loso_observations.csv"
    config = {
        "dataset": {"name": "fake_loso"},
        "preprocessing": {"window_ms": 20.0, "step_ms": 20.0, "normalization": "none"},
        "loso": {
            "label_column": "condition",
            "group_column": "subject",
            "trial_filter": {"split": "main"},
            "decoder": "logistic",
            "emission_mode": "uncalibrated",
            "normalization_scope": "per_group",
            "max_iter": 500,
        },
        "outputs": {"summary_csv": str(summary), "observations_csv": str(observations), "provenance": False},
    }
    monkeypatch.setattr("neureptrace.loso_time_decode.load_config", lambda _path: config)
    monkeypatch.setattr("neureptrace.loso_time_decode.load_epoch_dataset_from_config", lambda *_args, **_kwargs: _fake_bush_like_dataset())

    results = run_loso_time_decode(tmp_path / "config.yml")
    obs = pd.read_csv(observations)

    assert summary.exists()
    assert set(results["outer_group"]) == {"s1", "s2", "s3", "s4"}
    assert results["temporal_mode"].unique().tolist() == ["same_time_loso"]
    assert obs["group"].isin({"s1", "s2", "s3", "s4"}).all()
    assert sorted(obs["outer_group"].unique().tolist()) == ["s1", "s2", "s3", "s4"]


def test_loso_time_decode_selects_source_validated_train_window(tmp_path: Path, monkeypatch):
    summary = tmp_path / "loso_selected.csv"
    source_scores = tmp_path / "source_scores.csv"
    config = {
        "dataset": {"name": "fake_loso"},
        "preprocessing": {"window_ms": 20.0, "step_ms": 20.0, "normalization": "none"},
        "loso": {
            "label_column": "condition",
            "group_column": "subject",
            "trial_filter": {"split": "main"},
            "decoder": "logistic",
            "emission_mode": "uncalibrated",
            "normalization_scope": "per_group",
            "source_select_top_k": 1,
            "source_select_inner_splits": 2,
            "source_select_metric": "balanced_accuracy",
            "max_iter": 500,
        },
        "outputs": {"summary_csv": str(summary), "source_window_scores_csv": str(source_scores), "provenance": False},
    }
    monkeypatch.setattr("neureptrace.loso_time_decode.load_config", lambda _path: config)
    monkeypatch.setattr("neureptrace.loso_time_decode.load_epoch_dataset_from_config", lambda *_args, **_kwargs: _fake_bush_like_dataset())

    results = run_loso_time_decode(tmp_path / "config.yml")
    scores = pd.read_csv(source_scores)

    assert results["temporal_mode"].unique().tolist() == ["source_selected_train_window_ensemble"]
    assert results["n_train_windows"].unique().tolist() == [1]
    assert results["selected_train_times"].unique().tolist() == ["0.005"]
    assert source_scores.exists()
    assert set(scores["outer_group"]) == {"s1", "s2", "s3", "s4"}
    assert scores.loc[scores.groupby("outer_group")["source_select_score"].idxmax(), "train_time"].round(6).tolist() == [0.005] * 4


def test_loso_time_decode_can_restrict_heldout_groups(tmp_path: Path, monkeypatch):
    summary = tmp_path / "loso_one_subject.csv"
    config = {
        "dataset": {"name": "fake_loso"},
        "preprocessing": {"window_ms": 20.0, "step_ms": 20.0, "normalization": "none"},
        "loso": {
            "label_column": "condition",
            "group_column": "subject",
            "test_groups": ["s3"],
            "decoder": "logistic",
            "emission_mode": "uncalibrated",
            "max_iter": 500,
        },
        "outputs": {"summary_csv": str(summary), "provenance": False},
    }
    monkeypatch.setattr("neureptrace.loso_time_decode.load_config", lambda _path: config)
    monkeypatch.setattr("neureptrace.loso_time_decode.load_epoch_dataset_from_config", lambda *_args, **_kwargs: _fake_bush_like_dataset())

    results = run_loso_time_decode(tmp_path / "config.yml")

    assert results["outer_group"].unique().tolist() == ["s3"]
    assert results["n_test"].unique().tolist() == [4]


@pytest.mark.parametrize(
    ("preprocessing", "loso", "message"),
    [
        ({}, {"max_iter": True}, "loso.max_iter"),
        ({}, {"max_iter": 100.5}, "loso.max_iter"),
        ({}, {"tune_hyperparameters": "sometimes"}, "loso.tune_hyperparameters"),
        ({}, {"tuning_cv_splits": 0}, "loso.tuning_cv_splits"),
        ({}, {"source_select_top_k": True}, "loso.source_select_top_k"),
        ({}, {"source_select_top_k": -1}, "loso.source_select_top_k"),
        ({}, {"source_select_inner_splits": 1}, "loso.source_select_inner_splits"),
        ({}, {"decode_window": [True, 0.03]}, "loso.decode_window"),
        ({}, {"decode_window": [0.03, 0.01]}, "loso.decode_window"),
        ({}, {"source_select_window": [0.0, float("inf")]}, "loso.source_select_window"),
        ({}, {"temporal_train_window": [float("nan"), 0.03]}, "loso.temporal_train_window"),
        ({"window_ms": True}, {}, "preprocessing.window_ms"),
        ({"step_ms": 0}, {}, "preprocessing.step_ms"),
    ],
)
def test_loso_time_decode_rejects_malformed_result_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preprocessing: dict,
    loso: dict,
    message: str,
) -> None:
    config = _loso_config(tmp_path, preprocessing=preprocessing, loso=loso)
    monkeypatch.setattr("neureptrace.loso_time_decode.load_config", lambda _path: config)
    monkeypatch.setattr("neureptrace.loso_time_decode.load_epoch_dataset_from_config", lambda *_args, **_kwargs: _fake_bush_like_dataset())

    with pytest.raises(ValueError, match=message):
        run_loso_time_decode(tmp_path / "config.yml")


def test_loso_time_decode_rejects_malformed_write_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _loso_config(tmp_path)
    monkeypatch.setattr("neureptrace.loso_time_decode.load_config", lambda _path: config)

    with pytest.raises(ValueError, match="write_provenance"):
        run_loso_time_decode(tmp_path / "config.yml", write_provenance="maybe")
