from pathlib import Path

import numpy as np
import pandas as pd

from neureptrace.decoding import DECODER_CHOICES, normalize_decoder_name
from neureptrace.mne_time_decode import run_time_resolved_decode


class FakeEpochs:
    def __init__(self, data: np.ndarray, times: np.ndarray, metadata: pd.DataFrame):
        self._data = data
        self.times = times
        self.metadata = metadata

    def __len__(self) -> int:
        return self._data.shape[0]

    def copy(self):
        return FakeEpochs(self._data.copy(), self.times.copy(), self.metadata.copy())

    def pick(self, picks: str):
        return self

    def crop(self, tmin: float | None = None, tmax: float | None = None):
        keep = np.ones(len(self.times), dtype=bool)
        if tmin is not None:
            keep &= self.times >= tmin
        if tmax is not None:
            keep &= self.times <= tmax
        self.times = self.times[keep]
        self._data = self._data[:, :, keep]
        return self

    def __getitem__(self, keep):
        return FakeEpochs(self._data[keep], self.times.copy(), self.metadata.loc[keep].reset_index(drop=True))

    def get_data(self, copy: bool = False):
        return self._data.copy() if copy else self._data


def test_run_time_resolved_decode_writes_probability_observations(tmp_path: Path, monkeypatch):
    rng = np.random.default_rng(13)
    labels = np.array(["animate", "inanimate"] * 4)
    data = rng.normal(size=(8, 1, 5))
    data[labels == "animate", 0, :] += 0.5
    metadata = pd.DataFrame({"condition": labels, "session": ["a", "a", "b", "b", "c", "c", "d", "d"]})
    epochs = FakeEpochs(data, np.array([0.00, 0.01, 0.02, 0.03, 0.04]), metadata)
    monkeypatch.setattr("neureptrace.mne_time_decode.mne.read_epochs", lambda *args, **kwargs: epochs)

    out = tmp_path / "decode.csv"
    observations_out = tmp_path / "observations.csv"
    confusion_out = tmp_path / "confusion.csv"
    per_class_out = tmp_path / "per_class.csv"

    results = run_time_resolved_decode(
        epochs_path=tmp_path / "sub-01_epo.fif",
        label_column="condition",
        out_path=out,
        n_splits=2,
        window_ms=20,
        step_ms=20,
        max_iter=2000,
        observation_out_path=observations_out,
        subject="sub-01",
        emission_mode="both",
        confusion_out_path=confusion_out,
        per_class_out_path=per_class_out,
    )

    observations = pd.read_csv(observations_out)
    confusion = pd.read_csv(confusion_out)
    per_class = pd.read_csv(per_class_out)

    assert {"top_2_accuracy", "top_3_accuracy", "mean_true_label_rank", "median_true_label_rank"}.issubset(results.columns)
    assert len(observations) == 32
    assert {
        "subject",
        "fold",
        "decoder",
        "emission_mode",
        "time",
        "sample_index",
        "sequence_id",
        "true_class",
        "predicted_class",
        "probability_true_class",
        "confidence",
        "true_label_rank",
        "true_label_score",
        "rank1_label",
        "rank1_class",
        "rank1_score",
        "rank2_label",
        "rank2_class",
        "rank2_score",
        "prob_class_0",
        "prob_class_1",
    }.issubset(observations.columns)
    assert observations["subject"].unique().tolist() == ["sub-01"]
    assert sorted(observations["emission_mode"].unique().tolist()) == ["calibrated", "uncalibrated"]
    assert observations[["prob_class_0", "prob_class_1"]].sum(axis=1).round(6).tolist() == [1.0] * 32
    assert observations["true_label_rank"].between(1, 2).all()
    assert observations["true_label_score"].between(0.0, 1.0).all()
    assert observations["rank1_score"].ge(observations["rank2_score"]).all()
    assert {"true_label", "predicted_label", "count"}.issubset(confusion.columns)
    assert confusion["count"].sum() == len(observations)
    assert {"true_label", "n_trials", "n_correct", "accuracy"}.issubset(per_class.columns)
    assert per_class["n_trials"].sum() == len(observations)


def test_mne_time_decode_exposes_classifier_registry_decoders():
    assert "correlation-prototype" in DECODER_CHOICES
    assert "multinomial-logistic" in DECODER_CHOICES
    assert "random-forest" in DECODER_CHOICES
    assert normalize_decoder_name("correlation_prototype") == "correlation-prototype"
    assert normalize_decoder_name("multiclass-svm-weighted") == "multiclass-svm-weighted"
    assert normalize_decoder_name("shrinkage-lda") == "shrinkage_lda"


def test_run_time_resolved_decode_supports_registry_decoder(tmp_path: Path, monkeypatch):
    rng = np.random.default_rng(29)
    labels = np.array(["animate", "inanimate"] * 4)
    data = rng.normal(size=(8, 1, 5))
    data[labels == "animate", 0, :] += 0.5
    metadata = pd.DataFrame({"condition": labels, "session": ["a", "a", "b", "b", "c", "c", "d", "d"]})
    epochs = FakeEpochs(data, np.array([0.00, 0.01, 0.02, 0.03, 0.04]), metadata)
    monkeypatch.setattr("neureptrace.mne_time_decode.mne.read_epochs", lambda *args, **kwargs: epochs)

    out = tmp_path / "decode_registry.csv"
    observations_out = tmp_path / "observations_registry.csv"

    results = run_time_resolved_decode(
        epochs_path=tmp_path / "sub-01_epo.fif",
        label_column="condition",
        out_path=out,
        n_splits=2,
        window_ms=20,
        step_ms=20,
        decoder="correlation-prototype",
        emission_mode="uncalibrated",
        observation_out_path=observations_out,
    )
    observations = pd.read_csv(observations_out)

    assert results["decoder"].unique().tolist() == ["correlation-prototype"]
    assert observations["decoder"].unique().tolist() == ["correlation-prototype"]
    assert observations[["prob_class_0", "prob_class_1"]].sum(axis=1).round(6).tolist() == [1.0] * len(observations)


def test_temporal_train_window_ensemble_writes_all_test_times(tmp_path: Path, monkeypatch):
    rng = np.random.default_rng(17)
    labels = np.array(["animate", "inanimate"] * 4)
    data = rng.normal(size=(8, 1, 5))
    data[labels == "animate", 0, :2] += 0.75
    metadata = pd.DataFrame({"condition": labels, "session": ["a", "a", "b", "b", "c", "c", "d", "d"]})
    epochs = FakeEpochs(data, np.array([0.00, 0.01, 0.02, 0.03, 0.04]), metadata)
    monkeypatch.setattr("neureptrace.mne_time_decode.mne.read_epochs", lambda *args, **kwargs: epochs)

    out = tmp_path / "decode_temporal_ensemble.csv"
    observations_out = tmp_path / "observations_temporal_ensemble.csv"

    results = run_time_resolved_decode(
        epochs_path=tmp_path / "sub-01_epo.fif",
        label_column="condition",
        out_path=out,
        n_splits=2,
        window_ms=20,
        step_ms=20,
        max_iter=2000,
        observation_out_path=observations_out,
        subject="sub-01",
        temporal_train_window=(0.0, 0.015),
    )

    observations = pd.read_csv(observations_out)

    assert results["temporal_mode"].unique().tolist() == ["train_window_ensemble"]
    assert results["n_train_windows"].unique().tolist() == [1]
    assert results["train_time"].unique().round(6).tolist() == [0.005]
    assert sorted(results["time"].round(6).unique().tolist()) == [0.005, 0.025]
    assert len(results) == 4

    assert observations["temporal_mode"].unique().tolist() == ["train_window_ensemble"]
    assert observations["n_train_windows"].unique().tolist() == [1]
    assert observations["train_time"].unique().round(6).tolist() == [0.005]
    assert sorted(observations["test_time"].round(6).unique().tolist()) == [0.005, 0.025]
    assert observations[["prob_class_0", "prob_class_1"]].sum(axis=1).round(6).tolist() == [1.0] * len(observations)


def test_nested_temporal_selection_uses_generic_metadata_and_writes_selected_times(tmp_path: Path, monkeypatch):
    rng = np.random.default_rng(41)
    labels = np.array(["animate", "inanimate"] * 8)
    data = rng.normal(scale=0.1, size=(16, 1, 7))
    data[labels == "animate", 0, 2:4] += 2.0
    data[labels == "inanimate", 0, 2:4] -= 2.0
    metadata = pd.DataFrame(
        {
            "condition": labels,
            "partition": [f"run-{index // 2}" for index in range(16)],
            "epoch_kind": ["main"] * 16,
        }
    )
    epochs = FakeEpochs(data, np.array([0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]), metadata)
    monkeypatch.setattr("neureptrace.mne_time_decode.mne.read_epochs", lambda *args, **kwargs: epochs)

    out = tmp_path / "decode_temporal_selected.csv"
    observations_out = tmp_path / "observations_temporal_selected.csv"

    results = run_time_resolved_decode(
        epochs_path=tmp_path / "sub-01_epo.fif",
        label_column="condition",
        group_column="partition",
        out_path=out,
        n_splits=2,
        window_ms=20,
        step_ms=20,
        max_iter=2000,
        observation_out_path=observations_out,
        temporal_selection_window=(0.0, 0.05),
        temporal_selection_cv_splits=2,
        temporal_selection_top_k=2,
    )
    observations = pd.read_csv(observations_out)

    assert results["temporal_mode"].unique().tolist() == ["nested_train_window_selection"]
    assert results["temporal_selection_metric"].unique().tolist() == ["accuracy"]
    assert results["temporal_selection_top_k"].unique().tolist() == [2]
    assert results["n_train_windows"].unique().tolist() == [2]
    assert sorted(results["time"].round(6).unique().tolist()) == [0.005, 0.025, 0.045]
    assert results["temporal_selected_train_times"].str.contains("0.025").any()
    assert results["temporal_selection_scores"].str.len().min() > 0

    assert observations["temporal_mode"].unique().tolist() == ["nested_train_window_selection"]
    assert observations["temporal_selection_metric"].unique().tolist() == ["accuracy"]
    assert observations["group"].str.startswith("run-").all()
    assert observations[["prob_class_0", "prob_class_1"]].sum(axis=1).round(6).tolist() == [1.0] * len(observations)


def test_temporal_train_and_selection_windows_are_mutually_exclusive(tmp_path: Path, monkeypatch):
    rng = np.random.default_rng(43)
    labels = np.array(["animate", "inanimate"] * 4)
    data = rng.normal(size=(8, 1, 5))
    metadata = pd.DataFrame({"condition": labels})
    epochs = FakeEpochs(data, np.array([0.00, 0.01, 0.02, 0.03, 0.04]), metadata)
    monkeypatch.setattr("neureptrace.mne_time_decode.mne.read_epochs", lambda *args, **kwargs: epochs)

    try:
        run_time_resolved_decode(
            epochs_path=tmp_path / "sub-01_epo.fif",
            label_column="condition",
            out_path=tmp_path / "decode.csv",
            n_splits=2,
            window_ms=20,
            step_ms=20,
            temporal_train_window=(0.0, 0.03),
            temporal_selection_window=(0.0, 0.03),
        )
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("Expected temporal_train_window and temporal_selection_window to be mutually exclusive")


def test_run_time_resolved_decode_can_tune_decoder_hyperparameters(tmp_path: Path, monkeypatch):
    rng = np.random.default_rng(13)
    labels = np.array(["animate", "inanimate"] * 6)
    data = rng.normal(size=(12, 1, 5))
    data[labels == "animate", 0, :] += 0.5
    metadata = pd.DataFrame({"condition": labels, "session": ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e", "f", "f"]})
    epochs = FakeEpochs(data, np.array([0.00, 0.01, 0.02, 0.03, 0.04]), metadata)
    monkeypatch.setattr("neureptrace.mne_time_decode.mne.read_epochs", lambda *args, **kwargs: epochs)

    out = tmp_path / "decode.csv"
    observations_out = tmp_path / "observations.csv"

    results = run_time_resolved_decode(
        epochs_path=tmp_path / "sub-01_epo.fif",
        label_column="condition",
        out_path=out,
        n_splits=2,
        window_ms=20,
        step_ms=20,
        max_iter=2000,
        observation_out_path=observations_out,
        tune_hyperparameters=True,
        tuning_cv_splits=2,
        tuning_c_grid=(0.1, 1.0),
    )
    observations = pd.read_csv(observations_out)

    assert "best_params" in results.columns
    assert results["tuned_hyperparameters"].tolist() == [True] * len(results)
    assert observations["best_params"].str.contains("logisticregression__C", regex=False).all()
    assert observations["model_hash"].nunique() >= 1


def test_temporal_train_window_ensemble_can_tune_hyperparameters(tmp_path: Path, monkeypatch):
    rng = np.random.default_rng(19)
    labels = np.array(["animate", "inanimate"] * 6)
    data = rng.normal(size=(12, 1, 5))
    data[labels == "animate", 0, :2] += 0.75
    metadata = pd.DataFrame({"condition": labels, "session": ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e", "f", "f"]})
    epochs = FakeEpochs(data, np.array([0.00, 0.01, 0.02, 0.03, 0.04]), metadata)
    monkeypatch.setattr("neureptrace.mne_time_decode.mne.read_epochs", lambda *args, **kwargs: epochs)

    out = tmp_path / "decode_temporal_tuned.csv"
    observations_out = tmp_path / "observations_temporal_tuned.csv"

    results = run_time_resolved_decode(
        epochs_path=tmp_path / "sub-01_epo.fif",
        label_column="condition",
        out_path=out,
        n_splits=2,
        window_ms=20,
        step_ms=20,
        max_iter=2000,
        observation_out_path=observations_out,
        temporal_train_window=(0.0, 0.03),
        tune_hyperparameters=True,
        tuning_cv_splits=2,
        tuning_c_grid=(0.1, 1.0),
    )
    observations = pd.read_csv(observations_out)

    assert results["temporal_mode"].unique().tolist() == ["train_window_ensemble"]
    assert results["n_train_windows"].unique().tolist() == [2]
    assert results["tuned_hyperparameters"].tolist() == [True] * len(results)
    assert results["temporal_train_window_start"].unique().tolist() == [0.0]
    assert results["temporal_train_window_stop"].unique().tolist() == [0.03]
    assert observations["best_params"].str.contains("logisticregression__C", regex=False).all()
    assert observations["model_hash"].nunique() >= 1
