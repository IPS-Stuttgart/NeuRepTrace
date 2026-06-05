from pathlib import Path

import numpy as np
import pandas as pd

from neureptrace.decoding import DECODER_CHOICES, normalize_decoder_name
from neureptrace.mne_time_decode import (
    _apply_class_prior_correction,
    _align_probability_columns,
    _filter_splits_for_outer_test_groups,
    _shuffle_training_labels,
    apply_source_probability_calibration,
    fit_source_probability_calibrator,
    normalize_class_prior_correction,
    normalize_source_calibration,
    normalize_source_time_selection,
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


class RecordingFeatureDecoder:
    fit_feature_maxima: list[float] = []

    def fit(self, features: np.ndarray, labels: np.ndarray):
        self.classes_ = np.unique(labels)
        self.fit_feature_maxima.append(float(np.max(features)))
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probabilities = np.full((features.shape[0], len(self.classes_)), 0.2 / max(len(self.classes_) - 1, 1))
        dominant = (features[:, 0].astype(int) % len(self.classes_)).reshape(-1)
        for row_index, class_index in enumerate(dominant):
            probabilities[row_index, class_index] = 0.8
        return probabilities


def test_label_shuffle_helper_is_deterministic_and_count_preserving():
    labels = np.array([0, 0, 0, 1, 1, 2, 2, 2])

    shuffled_a = _shuffle_training_labels(labels, seed=7, context=("outer", "fold"))
    shuffled_b = _shuffle_training_labels(labels, seed=7, context=("outer", "fold"))
    shuffled_c = _shuffle_training_labels(labels, seed=8, context=("outer", "fold"))

    np.testing.assert_array_equal(shuffled_a, shuffled_b)
    assert sorted(shuffled_a.tolist()) == sorted(labels.tolist())
    assert not np.array_equal(shuffled_a, shuffled_c)


def test_outer_test_group_filter_preserves_fold_ids_and_accepts_subject_aliases():
    groups = np.array(["sub-01", "sub-01", "sub-02", "sub-02", "sub-03", "sub-03"])
    splits = [
        (0, (np.array([2, 3, 4, 5]), np.array([0, 1]))),
        (1, (np.array([0, 1, 4, 5]), np.array([2, 3]))),
        (2, (np.array([0, 1, 2, 3]), np.array([4, 5]))),
    ]

    selected = _filter_splits_for_outer_test_groups(splits, groups, ("1", "sub-03"))

    assert [fold for fold, _ in selected] == [0, 2]
    np.testing.assert_array_equal(selected[0][1][1], np.array([0, 1]))
    np.testing.assert_array_equal(selected[1][1][1], np.array([4, 5]))


def test_class_prior_correction_rebalances_train_fold_priors():
    probabilities = np.array([[0.6, 0.4], [0.4, 0.6]])
    train_labels = np.array([0, 0, 0, 1])

    corrected = _apply_class_prior_correction(
        probabilities,
        train_labels,
        classes=np.array([0, 1]),
        mode="train-uniform",
    )

    assert normalize_class_prior_correction("train-uniform") == "train_uniform"
    np.testing.assert_allclose(corrected.sum(axis=1), np.ones(2))
    assert corrected[0, 1] > corrected[0, 0]
    assert corrected[1, 1] > probabilities[1, 1]


def test_source_time_selection_uses_source_only_inner_validation(tmp_path: Path, monkeypatch):
    labels = np.array([0, 1, 2] * 3)
    groups = np.repeat(["sub-01", "sub-02", "sub-03"], 3)
    times = np.array([0.088, 0.136, 0.184, 0.232, 0.280])
    data = np.zeros((len(labels), 1, len(times)), dtype=float)
    good_time_index = 2
    for trial_index, label in enumerate(labels):
        data[trial_index, 0, :] = (label + 1) % 3
        data[trial_index, 0, good_time_index] = label
    metadata = pd.DataFrame({"condition": labels, "group": groups})
    epochs = FakeEpochs(data, times, metadata)

    monkeypatch.setattr("neureptrace.mne_time_decode.mne.read_epochs", lambda *args, **kwargs: epochs)
    monkeypatch.setattr("neureptrace.mne_time_decode.make_decoder", lambda *args, **kwargs: RecordingFeatureDecoder())

    out = tmp_path / "source_time.csv"
    observations_out = tmp_path / "source_time_observations.csv"
    results = run_time_resolved_decode(
        epochs_path=tmp_path / "sub-01_epo.fif",
        label_column="condition",
        group_column="group",
        outer_test_groups=("sub-01",),
        out_path=out,
        n_splits=3,
        window_ms=1,
        step_ms=48,
        decoder="logistic",
        emission_mode="uncalibrated",
        source_time_selection="source-oof-best-time",
        source_time_selection_times=(0.088, 0.136, 0.184, 0.232, 0.280),
        source_time_selection_output_time=0.184,
        observation_out_path=observations_out,
    )
    observations = pd.read_csv(observations_out)

    assert normalize_source_time_selection("source-oof-best-time") == "source_oof_best_time"
    assert results["decoder"].unique().tolist() == ["logistic_source_oof_best_time"]
    assert results["time"].unique().tolist() == [0.184]
    assert results["source_time_selection_selected_time"].round(3).tolist() == [0.184]
    assert observations["group"].unique().tolist() == ["sub-01"]
    assert observations["source_time_selection"].unique().tolist() == ["source_oof_best_time"]
    assert observations["source_time_selection_candidate_times"].str.contains("0.184").all()


def test_source_time_weighted_logits_writes_weights(tmp_path: Path, monkeypatch):
    labels = np.array([0, 1, 2] * 3)
    groups = np.repeat(["sub-01", "sub-02", "sub-03"], 3)
    times = np.array([0.088, 0.184])
    data = np.zeros((len(labels), 1, len(times)), dtype=float)
    for trial_index, label in enumerate(labels):
        data[trial_index, 0, 0] = (label + 1) % 3
        data[trial_index, 0, 1] = label
    metadata = pd.DataFrame({"condition": labels, "group": groups})
    epochs = FakeEpochs(data, times, metadata)

    monkeypatch.setattr("neureptrace.mne_time_decode.mne.read_epochs", lambda *args, **kwargs: epochs)
    monkeypatch.setattr("neureptrace.mne_time_decode.make_decoder", lambda *args, **kwargs: RecordingFeatureDecoder())

    results = run_time_resolved_decode(
        epochs_path=tmp_path / "sub-01_epo.fif",
        label_column="condition",
        group_column="group",
        outer_test_groups=("sub-01",),
        out_path=tmp_path / "source_time_weighted.csv",
        n_splits=3,
        window_ms=1,
        step_ms=96,
        decoder="logistic",
        emission_mode="uncalibrated",
        source_time_selection="source_oof_time_weighted_logits",
        source_time_selection_times=(0.088, 0.184),
    )

    assert results["decoder"].unique().tolist() == ["logistic_source_oof_time_weighted_logits"]
    assert results["source_time_selection_weights"].str.contains("|", regex=False).all()
    assert results["balanced_accuracy"].between(0.0, 1.0).all()


def test_source_time_classwise_weighted_logits_writes_matrix_weights(tmp_path: Path, monkeypatch):
    labels = np.array([0, 1, 2] * 3)
    groups = np.repeat(["sub-01", "sub-02", "sub-03"], 3)
    times = np.array([0.088, 0.184])
    data = np.zeros((len(labels), 1, len(times)), dtype=float)
    for trial_index, label in enumerate(labels):
        data[trial_index, 0, 0] = (label + 1) % 3
        data[trial_index, 0, 1] = label
    metadata = pd.DataFrame({"condition": labels, "group": groups})
    epochs = FakeEpochs(data, times, metadata)

    monkeypatch.setattr("neureptrace.mne_time_decode.mne.read_epochs", lambda *args, **kwargs: epochs)
    monkeypatch.setattr("neureptrace.mne_time_decode.make_decoder", lambda *args, **kwargs: RecordingFeatureDecoder())

    results = run_time_resolved_decode(
        epochs_path=tmp_path / "sub-01_epo.fif",
        label_column="condition",
        group_column="group",
        outer_test_groups=("sub-01",),
        out_path=tmp_path / "source_time_classwise.csv",
        n_splits=3,
        window_ms=1,
        step_ms=96,
        decoder="logistic",
        emission_mode="uncalibrated",
        source_time_selection="source-oof-classwise-time-weighted-logits",
        source_time_selection_times=(0.088, 0.184),
    )

    assert normalize_source_time_selection("source-oof-classwise-time-weighted-logits") == (
        "source_oof_classwise_time_weighted_logits"
    )
    assert results["decoder"].unique().tolist() == ["logistic_source_oof_classwise_time_weighted_logits"]
    assert results["source_time_selection_weight_type"].unique().tolist() == ["classwise"]
    assert results["source_time_selection_weights"].str.contains("/", regex=False).all()
    assert results["source_time_selection_weights"].str.split("/").map(len).unique().tolist() == [3]
    assert results["balanced_accuracy"].between(0.0, 1.0).all()


def test_source_calibrator_fits_deterministic_re_ranking():
    probabilities = np.array(
        [
            [0.70, 0.20, 0.10],
            [0.65, 0.25, 0.10],
            [0.62, 0.28, 0.10],
            [0.60, 0.15, 0.25],
            [0.58, 0.12, 0.30],
            [0.55, 0.15, 0.30],
        ]
    )
    labels = np.array([0, 1, 1, 2, 2, 2])

    calibrator = fit_source_probability_calibrator(probabilities, labels, "class-bias")
    corrected = apply_source_probability_calibration(probabilities, calibrator)

    assert normalize_source_calibration("temperature-plus-class-bias") == "temperature_plus_class_bias"
    assert calibrator.mode == "class_bias"
    assert calibrator.parameter
    np.testing.assert_allclose(corrected.sum(axis=1), np.ones(len(corrected)))
    assert corrected.argmax(axis=1).tolist().count(0) < probabilities.argmax(axis=1).tolist().count(0)


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


def test_source_calibration_fits_only_outer_train_subjects(tmp_path: Path, monkeypatch):
    RecordingFeatureDecoder.fit_feature_maxima = []
    labels = np.array(["a", "b", "a", "b", "a", "b", "a", "b"])
    data = np.repeat(np.arange(8, dtype=float).reshape(8, 1, 1), 2, axis=2)
    data[6:, :, :] = 100.0
    metadata = pd.DataFrame({"condition": labels, "session": ["s1", "s1", "s2", "s2", "s3", "s3", "target", "target"]})
    epochs = FakeEpochs(data, np.array([0.00, 0.01]), metadata)
    train_idx = np.array([0, 1, 2, 3, 4, 5])
    test_idx = np.array([6, 7])
    inner_splits = [
        (np.array([0, 1, 2, 3]), np.array([4, 5])),
        (np.array([0, 1, 4, 5]), np.array([2, 3])),
        (np.array([2, 3, 4, 5]), np.array([0, 1])),
    ]

    monkeypatch.setattr("neureptrace.mne_time_decode.mne.read_epochs", lambda *args, **kwargs: epochs)
    monkeypatch.setattr(
        "neureptrace.mne_time_decode.make_cross_validator",
        lambda labels, groups, n_splits: iter([(train_idx, test_idx)]),
    )
    monkeypatch.setattr(
        "neureptrace.mne_time_decode.make_tuning_cross_validator",
        lambda labels, groups, n_splits: inner_splits,
    )
    monkeypatch.setattr("neureptrace.mne_time_decode.make_decoder", lambda *args, **kwargs: RecordingFeatureDecoder())

    out = tmp_path / "decode_source_calibration.csv"
    observations_out = tmp_path / "observations_source_calibration.csv"

    results = run_time_resolved_decode(
        epochs_path=tmp_path / "sub-01_epo.fif",
        label_column="condition",
        group_column="session",
        out_path=out,
        n_splits=2,
        window_ms=10,
        step_ms=10,
        emission_mode="uncalibrated",
        observation_out_path=observations_out,
        time_decode_backend="sklearn",
        source_calibration="class_bias",
    )
    observations = pd.read_csv(observations_out)

    assert results["source_calibration"].unique().tolist() == ["class_bias"]
    assert observations["source_calibration"].unique().tolist() == ["class_bias"]
    assert sorted(observations["true_class"].unique().tolist()) == ["a", "b"]
    assert observations["sample_index"].unique().tolist() == [6, 7]
    assert RecordingFeatureDecoder.fit_feature_maxima
    assert max(RecordingFeatureDecoder.fit_feature_maxima) < 100.0


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


def test_run_time_resolved_decode_can_restrict_decode_window(tmp_path: Path, monkeypatch):
    rng = np.random.default_rng(19)
    labels = np.array(["animate", "inanimate"] * 4)
    data = rng.normal(size=(8, 1, 6))
    data[labels == "animate", 0, 2:4] += 0.5
    metadata = pd.DataFrame({"condition": labels, "session": ["a", "a", "b", "b", "c", "c", "d", "d"]})
    epochs = FakeEpochs(data, np.array([0.00, 0.01, 0.02, 0.03, 0.04, 0.05]), metadata)
    monkeypatch.setattr("neureptrace.mne_time_decode.mne.read_epochs", lambda *args, **kwargs: epochs)

    out = tmp_path / "decode_window.csv"
    observations_out = tmp_path / "observations_decode_window.csv"

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
        decode_window=(0.02, 0.03),
    )
    observations = pd.read_csv(observations_out)

    assert results["time"].round(6).unique().tolist() == [0.025]
    assert observations["time"].round(6).unique().tolist() == [0.025]


def test_foldlocal_time_resolved_decode_can_restrict_decode_window(tmp_path: Path, monkeypatch):
    from neureptrace.mne_time_decode_foldlocal import run_time_resolved_decode as run_foldlocal_decode

    rng = np.random.default_rng(31)
    labels = np.array(["animate", "inanimate"] * 4)
    data = rng.normal(size=(8, 1, 6))
    data[labels == "animate", 0, 2:4] += 0.5
    metadata = pd.DataFrame({"condition": labels, "session": ["a", "a", "b", "b", "c", "c", "d", "d"]})
    epochs = FakeEpochs(data, np.array([0.00, 0.01, 0.02, 0.03, 0.04, 0.05]), metadata)
    monkeypatch.setattr("neureptrace.mne_time_decode.mne.read_epochs", lambda *args, **kwargs: epochs)

    out = tmp_path / "foldlocal_decode_window.csv"
    observations_out = tmp_path / "foldlocal_observations_decode_window.csv"

    results = run_foldlocal_decode(
        epochs_path=tmp_path / "sub-01_epo.fif",
        label_column="condition",
        group_column="session",
        out_path=out,
        n_splits=2,
        window_ms=20,
        step_ms=20,
        emission_mode="uncalibrated",
        observation_out_path=observations_out,
        decode_window=(0.02, 0.03),
        source_time_selection="none",
        source_time_selection_times=(0.088, 0.184),
        source_time_selection_output_time=0.184,
    )
    observations = pd.read_csv(observations_out)

    assert results["time"].round(6).unique().tolist() == [0.025]
    assert observations["time"].round(6).unique().tolist() == [0.025]

    source_time_results = run_foldlocal_decode(
        epochs_path=tmp_path / "sub-01_epo.fif",
        label_column="condition",
        group_column="session",
        out_path=tmp_path / "foldlocal_source_time.csv",
        n_splits=2,
        window_ms=20,
        step_ms=20,
        emission_mode="uncalibrated",
        decode_window=(0.02, 0.03),
        source_time_selection="source_oof_best_time",
        source_time_selection_times=(0.025,),
        source_time_selection_output_time=0.025,
    )
    assert source_time_results["temporal_mode"].unique().tolist() == ["source_oof_best_time"]
    assert source_time_results["time"].round(6).unique().tolist() == [0.025]
    assert source_time_results["source_time_selection_normalization_scope"].unique().tolist() == ["inner_train_fold"]


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
    assert "multinomial-logistic-weighted" in DECODER_CHOICES
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
