from pathlib import Path

import numpy as np
import pandas as pd

from neureptrace.decoding import DECODER_CHOICES, normalize_decoder_name
from neureptrace.mne_time_decode import (
    _align_probability_columns,
    _shuffle_training_labels,
    normalize_time_decode_backend,
    run_time_resolved_decode,
)


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


class MissingClassDecoder:
    def fit(self, features: np.ndarray, labels: np.ndarray):
        self.classes_ = np.unique(labels)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probabilities = np.tile(np.array([[0.7, 0.3]]), (features.shape[0], 1))
        return probabilities[:, : len(self.classes_)]


class RecordingDecoder:
    fit_labels: list[np.ndarray] = []

    def fit(self, features: np.ndarray, labels: np.ndarray):
        self.classes_ = np.unique(labels)
        self.fit_labels.append(np.asarray(labels, dtype=int).copy())
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.full((features.shape[0], len(self.classes_)), 1.0 / len(self.classes_))


def test_label_shuffle_helper_is_deterministic_and_count_preserving():
    labels = np.array([0, 0, 0, 1, 1, 2, 2, 2])

    shuffled_a = _shuffle_training_labels(labels, seed=7, context=("outer", "fold"))
    shuffled_b = _shuffle_training_labels(labels, seed=7, context=("outer", "fold"))
    shuffled_c = _shuffle_training_labels(labels, seed=8, context=("outer", "fold"))

    np.testing.assert_array_equal(shuffled_a, shuffled_b)
    assert sorted(shuffled_a.tolist()) == sorted(labels.tolist())
    assert not np.array_equal(shuffled_a, shuffled_c)


def test_align_probability_columns_expands_missing_model_classes():
    model = MissingClassDecoder()
    model.classes_ = np.array([2, 0])
    probabilities = np.array([[0.75, 0.25], [0.1, 0.9]])

    aligned = _align_probability_columns(probabilities, model=model, classes=np.array([0, 1, 2]))

    np.testing.assert_allclose(
        aligned,
        np.array(
            [
                [0.25, 0.0, 0.75],
                [0.9, 0.0, 0.1],
            ]
        ),
    )
    np.testing.assert_allclose(aligned.sum(axis=1), np.ones(2))


def test_run_time_resolved_decode_aligns_missing_fold_class_probabilities(tmp_path: Path, monkeypatch):
    labels = np.array(["a", "b", "c", "a", "b", "c"])
    data = np.arange(12, dtype=float).reshape(6, 1, 2)
    metadata = pd.DataFrame({"condition": labels, "session": ["s1", "s1", "s1", "s2", "s2", "s2"]})
    epochs = FakeEpochs(data, np.array([0.00, 0.01]), metadata)
    monkeypatch.setattr("neureptrace.mne_time_decode.mne.read_epochs", lambda *args, **kwargs: epochs)
    monkeypatch.setattr(
        "neureptrace.mne_time_decode.make_cross_validator",
        lambda labels, groups, n_splits: iter([(np.array([0, 2, 3, 5]), np.array([1, 4]))]),
    )
    monkeypatch.setattr("neureptrace.mne_time_decode.make_decoder", lambda *args, **kwargs: MissingClassDecoder())

    out = tmp_path / "decode_missing_class.csv"
    observations_out = tmp_path / "observations_missing_class.csv"

    results = run_time_resolved_decode(
        epochs_path=tmp_path / "sub-01_epo.fif",
        label_column="condition",
        out_path=out,
        n_splits=2,
        window_ms=10,
        step_ms=10,
        emission_mode="uncalibrated",
        observation_out_path=observations_out,
        time_decode_backend="sklearn",
    )
    observations = pd.read_csv(observations_out)

    assert results["n_classes"].unique().tolist() == [3]
    assert {"prob_class_0", "prob_class_1", "prob_class_2"}.issubset(observations.columns)
    assert observations["true_class"].unique().tolist() == ["b"]
    assert observations["prob_class_1"].tolist() == [0.0] * len(observations)
    assert observations["probability_true_class"].tolist() == [0.0] * len(observations)
    assert observations[["prob_class_0", "prob_class_1", "prob_class_2"]].sum(axis=1).round(6).tolist() == [1.0] * len(observations)


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

    run_time_resolved_decode(
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
    )

    observations = pd.read_csv(observations_out)

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
        "prob_class_0",
        "prob_class_1",
    }.issubset(observations.columns)
    assert observations["subject"].unique().tolist() == ["sub-01"]
    assert sorted(observations["emission_mode"].unique().tolist()) == ["calibrated", "uncalibrated"]
    assert observations[["prob_class_0", "prob_class_1"]].sum(axis=1).round(6).tolist() == [1.0] * 32


def test_run_time_resolved_decode_label_shuffle_keeps_test_labels_and_marks_outputs(tmp_path: Path, monkeypatch):
    RecordingDecoder.fit_labels = []
    labels = np.array(["animate", "animate", "inanimate", "inanimate", "animate", "inanimate"])
    data = np.arange(30, dtype=float).reshape(6, 1, 5)
    metadata = pd.DataFrame({"condition": labels, "session": ["a", "a", "b", "b", "c", "c"]})
    epochs = FakeEpochs(data, np.array([0.00, 0.01, 0.02, 0.03, 0.04]), metadata)
    train_idx = np.array([0, 1, 2, 3])
    test_idx = np.array([4, 5])
    monkeypatch.setattr("neureptrace.mne_time_decode.mne.read_epochs", lambda *args, **kwargs: epochs)
    monkeypatch.setattr(
        "neureptrace.mne_time_decode.make_cross_validator",
        lambda labels, groups, n_splits: iter([(train_idx, test_idx)]),
    )
    monkeypatch.setattr("neureptrace.mne_time_decode.make_decoder", lambda *args, **kwargs: RecordingDecoder())

    out = tmp_path / "decode_shuffle.csv"
    observations_out = tmp_path / "observations_shuffle.csv"

    results = run_time_resolved_decode(
        epochs_path=tmp_path / "sub-01_epo.fif",
        label_column="condition",
        out_path=out,
        n_splits=2,
        window_ms=20,
        step_ms=20,
        emission_mode="uncalibrated",
        observation_out_path=observations_out,
        time_decode_backend="sklearn",
        label_shuffle_control=True,
        label_shuffle_seed=7,
    )
    observations = pd.read_csv(observations_out)

    assert results["label_shuffle_control"].unique().tolist() == [True]
    assert results["label_shuffle_seed"].unique().tolist() == [7]
    assert observations["label_shuffle_control"].unique().tolist() == [True]
    assert observations["label_shuffle_seed"].unique().tolist() == [7]
    assert observations.sort_values("sample_index")["true_class"].unique().tolist() == ["animate", "inanimate"]
    assert len(RecordingDecoder.fit_labels) == 2
    assert all(sorted(fit_labels.tolist()) == [0, 0, 1, 1] for fit_labels in RecordingDecoder.fit_labels)
    np.testing.assert_array_equal(RecordingDecoder.fit_labels[0], RecordingDecoder.fit_labels[1])


def test_normalize_time_decode_backend_accepts_mne_alias():
    assert normalize_time_decode_backend(None) == "auto"
    assert normalize_time_decode_backend("mne-decoding") == "mne"


def test_mne_sliding_backend_matches_existing_same_time_decode(tmp_path: Path, monkeypatch):
    rng = np.random.default_rng(41)
    labels = np.array(["animate", "inanimate"] * 8)
    data = rng.normal(size=(16, 2, 6))
    data[labels == "animate", 0, 2:5] += 0.7
    metadata = pd.DataFrame({"condition": labels, "session": np.repeat(["a", "b", "c", "d"], 4)})
    epochs = FakeEpochs(data, np.array([0.00, 0.01, 0.02, 0.03, 0.04, 0.05]), metadata)
    monkeypatch.setattr("neureptrace.mne_time_decode.mne.read_epochs", lambda *args, **kwargs: epochs)

    sklearn_out = tmp_path / "decode_sklearn.csv"
    mne_out = tmp_path / "decode_mne.csv"
    sklearn_observations_out = tmp_path / "observations_sklearn.csv"
    mne_observations_out = tmp_path / "observations_mne.csv"

    sklearn_results = run_time_resolved_decode(
        epochs_path=tmp_path / "sub-01_epo.fif",
        label_column="condition",
        out_path=sklearn_out,
        n_splits=2,
        window_ms=20,
        step_ms=20,
        max_iter=2000,
        emission_mode="calibrated",
        observation_out_path=sklearn_observations_out,
        time_decode_backend="sklearn",
    )
    mne_results = run_time_resolved_decode(
        epochs_path=tmp_path / "sub-01_epo.fif",
        label_column="condition",
        out_path=mne_out,
        n_splits=2,
        window_ms=20,
        step_ms=20,
        max_iter=2000,
        emission_mode="calibrated",
        observation_out_path=mne_observations_out,
        time_decode_backend="mne",
    )

    sort_result_columns = ["fold", "time", "emission_mode"]
    sklearn_results = sklearn_results.sort_values(sort_result_columns).reset_index(drop=True)
    mne_results = mne_results.sort_values(sort_result_columns).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        sklearn_results[["fold", "time", "accuracy", "log_loss", "brier", "ece"]],
        mne_results[["fold", "time", "accuracy", "log_loss", "brier", "ece"]],
        check_exact=False,
        atol=1e-12,
        rtol=1e-12,
    )

    sklearn_observations = pd.read_csv(sklearn_observations_out).sort_values(["fold", "time", "sample_index"]).reset_index(drop=True)
    mne_observations = pd.read_csv(mne_observations_out).sort_values(["fold", "time", "sample_index"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        sklearn_observations[["fold", "time", "sample_index", "prob_class_0", "prob_class_1"]],
        mne_observations[["fold", "time", "sample_index", "prob_class_0", "prob_class_1"]],
        check_exact=False,
        atol=1e-12,
        rtol=1e-12,
    )
    assert mne_observations["backend"].unique().tolist() == ["mne"]


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
