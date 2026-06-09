from pathlib import Path

import numpy as np
import pandas as pd

from neureptrace.decoding import DECODER_CHOICES, normalize_decoder_name
from neureptrace.decoding.source_alignment import SourceAlignmentResult
from neureptrace.mne_time_decode import (
    _apply_class_prior_correction,
    _align_probability_columns,
    _alignment_anchor_values,
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


def test_run_time_resolved_decode_applies_strict_alignment_with_shuffled_train_labels(tmp_path: Path, monkeypatch):
    labels = np.tile(np.array([0, 1, 0, 1]), 3)
    groups = np.repeat(["sub-01", "sub-02", "sub-03"], 4)
    times = np.array([0.180, 0.184, 0.188])
    data = np.zeros((len(labels), 2, len(times)), dtype=float)
    for trial_index, label in enumerate(labels):
        data[trial_index, :, :] = label
    metadata = pd.DataFrame({"condition": labels, "group": groups})
    epochs = FakeEpochs(data, times, metadata)
    alignment_train_labels = []
    alignment_configs = []

    def fake_align_train_test_features(**kwargs):
        assert kwargs.get("target_labels") is None
        alignment_train_labels.append(np.asarray(kwargs["train_labels"], dtype=int).copy())
        alignment_configs.append(kwargs["config"])
        return SourceAlignmentResult(
            train_features=np.asarray(kwargs["train_features"], dtype=float),
            test_features=np.asarray(kwargs["test_features"], dtype=float),
            metadata={
                "alignment_method": "procrustes",
                "alignment_anchor_mode": "class_mean",
                "alignment_target_projection": "group_projection",
                "alignment_n_components": 2,
            },
            diagnostics={
                "alignment_method": "procrustes",
                "sample_mode": "class_mean",
                "n_source_subjects": 2,
                "n_classes": 2,
                "n_alignment_rows": 2,
                "n_repetitions_per_class": "",
                "requested_components": 2,
                "actual_components": 2,
                "feature_dim": 2,
                "decode_feature_dim": 2,
                "uses_channel_projection_collapse": False,
                "alignment_dimensionality_reduction": False,
                "anchor_row_correlation_before": 0.1,
                "anchor_row_correlation_after": 0.9,
                "source_inner_decoding_before_alignment": 0.5,
                "source_inner_decoding_after_alignment": 0.75,
                "source_inner_raw_balanced_accuracy": 0.5,
                "source_inner_aligned_balanced_accuracy": 0.75,
                "source_inner_aligned_minus_raw": 0.25,
                "source_inner_validation_type": "strict_source_loso_nearest_centroid_group_projection",
                "target_transform_type": "source_group_projection",
            },
        )

    monkeypatch.setattr("neureptrace.mne_time_decode.mne.read_epochs", lambda *args, **kwargs: epochs)
    monkeypatch.setattr("neureptrace.mne_time_decode.make_decoder", lambda *args, **kwargs: RecordingFeatureDecoder())
    monkeypatch.setattr("neureptrace.mne_time_decode.align_train_test_features", fake_align_train_test_features)

    results = run_time_resolved_decode(
        epochs_path=tmp_path / "synthetic-epo.fif",
        dataset_name="synthetic",
        label_column="condition",
        group_column="group",
        outer_test_groups=("sub-01",),
        out_path=tmp_path / "aligned.csv",
        n_splits=3,
        window_ms=1,
        step_ms=4,
        decoder="logistic",
        emission_mode="uncalibrated",
        time_decode_backend="sklearn",
        alignment_method="procrustes",
        alignment_times="same_decode_window",
        label_shuffle_control=True,
        label_shuffle_seed=13,
    )

    assert alignment_train_labels
    assert alignment_configs[0].same_decode_window is True
    unshuffled_train = labels[groups != "sub-01"]
    assert sorted(alignment_train_labels[0].tolist()) == sorted(unshuffled_train.tolist())
    assert not np.array_equal(alignment_train_labels[0], unshuffled_train)
    assert set(results["alignment_method"]) == {"procrustes"}
    assert set(results["alignment_target_projection"]) == {"group_projection"}
    diagnostics = pd.read_csv(tmp_path / "alignment_diagnostics.csv")
    assert diagnostics["dataset"].unique().tolist() == ["synthetic"]
    assert diagnostics["test_subject"].unique().tolist() == ["sub-01"]
    assert diagnostics["actual_components"].unique().tolist() == [2]
    assert diagnostics["feature_dim"].unique().tolist() == [2]
    assert diagnostics["decode_feature_dim"].unique().tolist() == [2]
    assert diagnostics["source_inner_raw_balanced_accuracy"].unique().tolist() == [0.5]
    assert diagnostics["source_inner_aligned_balanced_accuracy"].unique().tolist() == [0.75]
    assert diagnostics["source_inner_aligned_minus_raw"].unique().tolist() == [0.25]
    assert diagnostics["target_transform_type"].unique().tolist() == ["source_group_projection"]
    assert np.allclose(diagnostics["alignment_window_center"], diagnostics["decode_window_center"])
    assert np.allclose(diagnostics["alignment_window_size"], diagnostics["decode_window_size"])


def test_run_time_resolved_decode_passes_target_labels_for_oracle_alignment(tmp_path: Path, monkeypatch):
    labels = np.tile(np.array([0, 1, 0, 1]), 3)
    groups = np.repeat(["sub-01", "sub-02", "sub-03"], 4)
    times = np.array([0.180, 0.184, 0.188])
    data = np.zeros((len(labels), 2, len(times)), dtype=float)
    for trial_index, label in enumerate(labels):
        data[trial_index, :, :] = label
    metadata = pd.DataFrame({"condition": labels, "group": groups})
    epochs = FakeEpochs(data, times, metadata)
    target_label_calls = []

    def fake_align_train_test_features(**kwargs):
        assert kwargs.get("target_labels") is not None
        target_label_calls.append(np.asarray(kwargs["target_labels"], dtype=int).copy())
        return SourceAlignmentResult(
            train_features=np.asarray(kwargs["train_features"], dtype=float),
            test_features=np.asarray(kwargs["test_features"], dtype=float),
            metadata={
                "alignment_method": "procrustes",
                "alignment_anchor_mode": "class_mean",
                "alignment_target_projection": "oracle_target_calibrated_alignment",
                "alignment_oracle_target_calibrated": True,
                "alignment_debug_upper_bound": True,
                "alignment_valid_for_benchmark": False,
                "alignment_protocol_note": "debug upper bound only; not valid for benchmark",
                "alignment_n_components": 2,
            },
        )

    monkeypatch.setattr("neureptrace.mne_time_decode.mne.read_epochs", lambda *args, **kwargs: epochs)
    monkeypatch.setattr("neureptrace.mne_time_decode.make_decoder", lambda *args, **kwargs: RecordingFeatureDecoder())
    monkeypatch.setattr("neureptrace.mne_time_decode.align_train_test_features", fake_align_train_test_features)

    results = run_time_resolved_decode(
        epochs_path=tmp_path / "synthetic-epo.fif",
        label_column="condition",
        group_column="group",
        outer_test_groups=("sub-01",),
        out_path=tmp_path / "oracle_aligned.csv",
        n_splits=3,
        window_ms=1,
        step_ms=4,
        decoder="logistic",
        emission_mode="uncalibrated",
        time_decode_backend="sklearn",
        alignment_method="procrustes",
        alignment_target_projection="oracle_target_calibrated_alignment",
    )

    assert target_label_calls
    np.testing.assert_array_equal(target_label_calls[0], labels[groups == "sub-01"])
    assert set(results["alignment_target_projection"]) == {"oracle_target_calibrated_alignment"}
    assert set(results["alignment_valid_for_benchmark"]) == {False}


def test_run_time_resolved_decode_target_calibration_excludes_scored_rows(tmp_path: Path, monkeypatch):
    labels = np.tile(np.array([0, 1, 0, 1, 0, 1]), 3)
    groups = np.repeat(["sub-01", "sub-02", "sub-03"], 6)
    times = np.array([0.180, 0.184, 0.188])
    data = np.zeros((len(labels), 2, len(times)), dtype=float)
    for trial_index in range(len(labels)):
        data[trial_index, :, :] = trial_index
    metadata = pd.DataFrame({"condition": labels, "group": groups})
    epochs = FakeEpochs(data, times, metadata)
    scored_feature_rows = []
    calibration_feature_rows = []
    calibration_label_rows = []

    def fake_align_train_test_features(**kwargs):
        assert kwargs.get("target_labels") is None
        assert kwargs.get("target_anchor_values") is None
        assert kwargs.get("target_calibration_features") is not None
        assert kwargs.get("target_calibration_labels") is not None
        scored_feature_rows.append(np.asarray(kwargs["test_features"], dtype=float).copy())
        calibration_feature_rows.append(np.asarray(kwargs["target_calibration_features"], dtype=float).copy())
        calibration_label_rows.append(np.asarray(kwargs["target_calibration_labels"], dtype=int).copy())
        return SourceAlignmentResult(
            train_features=np.asarray(kwargs["train_features"], dtype=float),
            test_features=np.asarray(kwargs["test_features"], dtype=float),
            metadata={
                "alignment_method": "procrustes",
                "alignment_anchor_mode": "class_mean",
                "alignment_target_projection": "target_calibrated_alignment",
                "alignment_target_calibrated": True,
                "alignment_oracle_target_calibrated": False,
                "alignment_debug_upper_bound": False,
                "alignment_valid_for_benchmark": True,
                "alignment_target_alignment_rows": 2,
                "alignment_target_labels_used": True,
                "alignment_target_calibration_per_anchor": 1,
                "alignment_target_calibration_seed": 17,
                "alignment_n_components": 2,
            },
            diagnostics={
                "alignment_method": "procrustes",
                "sample_mode": "class_mean",
                "n_source_subjects": 2,
                "n_classes": 2,
                "n_alignment_rows": 2,
                "n_repetitions_per_class": "",
                "requested_components": 2,
                "actual_components": 2,
                "feature_dim": 2,
                "decode_feature_dim": 2,
                "uses_channel_projection_collapse": False,
                "alignment_dimensionality_reduction": False,
                "target_transform_type": "target_calibrated_template_procrustes",
            },
        )

    monkeypatch.setattr("neureptrace.mne_time_decode.mne.read_epochs", lambda *args, **kwargs: epochs)
    monkeypatch.setattr("neureptrace.mne_time_decode.make_decoder", lambda *args, **kwargs: RecordingFeatureDecoder())
    monkeypatch.setattr("neureptrace.mne_time_decode.align_train_test_features", fake_align_train_test_features)

    observations_out = tmp_path / "target_calibrated_observations.csv"
    results = run_time_resolved_decode(
        epochs_path=tmp_path / "synthetic-epo.fif",
        label_column="condition",
        group_column="group",
        outer_test_groups=("sub-01",),
        out_path=tmp_path / "target_calibrated.csv",
        n_splits=3,
        window_ms=1,
        step_ms=4,
        decoder="logistic",
        emission_mode="uncalibrated",
        time_decode_backend="sklearn",
        alignment_method="procrustes",
        alignment_target_projection="target_calibrated_alignment",
        alignment_target_calibration_per_anchor=1,
        alignment_target_calibration_seed=17,
        observation_out_path=observations_out,
    )
    observations = pd.read_csv(observations_out)

    assert scored_feature_rows
    assert {tuple(sorted(row.tolist())) for row in calibration_label_rows} == {(0, 1)}
    assert all(features.shape[0] == 4 for features in scored_feature_rows)
    assert all(features.shape[0] == 2 for features in calibration_feature_rows)
    for scored, calibrated in zip(scored_feature_rows, calibration_feature_rows, strict=True):
        assert set(scored[:, 0]).isdisjoint(set(calibrated[:, 0]))
    assert set(results["alignment_target_projection"]) == {"target_calibrated_alignment"}
    assert set(results["alignment_target_calibrated"]) == {True}
    assert set(results["alignment_valid_for_benchmark"]) == {True}
    assert set(results["n_test"]) == {4}
    assert len(observations) == 4 * len(results)
    diagnostics = pd.read_csv(tmp_path / "alignment_diagnostics.csv")
    assert diagnostics["alignment_target_projection"].unique().tolist() == ["target_calibrated_alignment"]
    assert diagnostics["target_transform_type"].unique().tolist() == ["target_calibrated_template_procrustes"]


def test_run_time_resolved_decode_skips_anchor_lookup_for_unsupervised_alignment(tmp_path: Path, monkeypatch):
    labels = np.tile(np.array([0, 1, 0, 1]), 3)
    groups = np.repeat(["sub-01", "sub-02", "sub-03"], 4)
    times = np.array([0.180, 0.184, 0.188])
    data = np.zeros((len(labels), 2, len(times)), dtype=float)
    metadata = pd.DataFrame({"condition": labels, "group": groups})
    epochs = FakeEpochs(data, times, metadata)

    def fake_align_train_test_features(**kwargs):
        assert kwargs["config"].method == "coral"
        assert kwargs.get("train_anchor_values") is None
        assert kwargs.get("target_anchor_values") is None
        return SourceAlignmentResult(
            train_features=np.asarray(kwargs["train_features"], dtype=float),
            test_features=np.asarray(kwargs["test_features"], dtype=float),
            metadata={
                "alignment_method": "coral",
                "alignment_anchor_mode": "stimulus_id_mean",
                "alignment_target_projection": "group_projection",
                "alignment_uses_unlabeled_target_data": True,
            },
            diagnostics={
                "alignment_method": "coral",
                "sample_mode": "unlabeled_covariance",
                "uses_unlabeled_target_data": True,
                "covariance_alignment_estimator": "diagonal",
                "target_transform_type": "unlabeled_target_covariance_recoloring",
            },
        )

    monkeypatch.setattr("neureptrace.mne_time_decode.mne.read_epochs", lambda *args, **kwargs: epochs)
    monkeypatch.setattr("neureptrace.mne_time_decode.make_decoder", lambda *args, **kwargs: RecordingFeatureDecoder())
    monkeypatch.setattr("neureptrace.mne_time_decode.align_train_test_features", fake_align_train_test_features)

    results = run_time_resolved_decode(
        epochs_path=tmp_path / "synthetic-epo.fif",
        label_column="condition",
        group_column="group",
        outer_test_groups=("sub-01",),
        out_path=tmp_path / "unsupervised_aligned.csv",
        n_splits=3,
        window_ms=1,
        step_ms=4,
        decoder="logistic",
        emission_mode="uncalibrated",
        time_decode_backend="sklearn",
        alignment_method="coral",
        alignment_anchor_mode="stimulus_id_mean",
    )

    assert set(results["alignment_method"]) == {"coral"}


def test_alignment_anchor_values_auto_select_ds000117_columns():
    metadata = pd.DataFrame(
        {
            "condition": ["Famous", "Famous", "Scrambled"],
            "stim_file": ["meg/f001.bmp", "meg/f001.bmp", "meg/s001.bmp"],
            "trigger": [5, 6, 17],
            "run": ["01", "01", "01"],
        }
    )
    labels = np.array([0, 0, 1])

    stimulus = _alignment_anchor_values(
        metadata,
        labels,
        label_column="condition",
        anchor_mode="stimulus_id_mean",
    )
    event = _alignment_anchor_values(
        metadata,
        labels,
        label_column="condition",
        anchor_mode="event_code_mean",
    )
    indexed = _alignment_anchor_values(
        metadata,
        labels,
        label_column="condition",
        anchor_mode="run_event_index_within_stimulus",
    )

    assert stimulus.column == "stim_file"
    assert stimulus.values.tolist() == ["meg/f001.bmp", "meg/f001.bmp", "meg/s001.bmp"]
    assert event.column == "trigger"
    assert event.values.tolist() == ["5", "6", "17"]
    assert indexed.column == "run+stim_file+event_index"
    assert indexed.values.tolist() == [
        "run=01|stimulus=meg/f001.bmp|label=Famous|index=1",
        "run=01|stimulus=meg/f001.bmp|label=Famous|index=2",
        "run=01|stimulus=meg/s001.bmp|label=Scrambled|index=1",
    ]


def test_alignment_anchor_values_prefer_ds000117_canonical_identity_columns():
    metadata = pd.DataFrame(
        {
            "condition": ["Famous", "Famous", "Scrambled"],
            "stimulus_id": ["f001", "f001", "s001"],
            "stim_file": ["meg/f001.bmp", "meg/f001.bmp", "meg/s001.bmp"],
            "event_code": ["5", "6", "17"],
            "trigger": [5, 6, 17],
            "run": ["01", "01", "01"],
        }
    )
    labels = np.array([0, 0, 1])

    stimulus = _alignment_anchor_values(
        metadata,
        labels,
        label_column="condition",
        anchor_mode="stimulus_id_mean",
    )
    event = _alignment_anchor_values(
        metadata,
        labels,
        label_column="condition",
        anchor_mode="event_code_mean",
    )
    indexed = _alignment_anchor_values(
        metadata,
        labels,
        label_column="condition",
        anchor_mode="run_event_index_within_stimulus",
    )

    assert stimulus.column == "stimulus_id"
    assert stimulus.values.tolist() == ["f001", "f001", "s001"]
    assert event.column == "event_code"
    assert event.values.tolist() == ["5", "6", "17"]
    assert indexed.column == "run+stimulus_id+event_index"
    assert indexed.values.tolist() == [
        "run=01|stimulus=f001|label=Famous|index=1",
        "run=01|stimulus=f001|label=Famous|index=2",
        "run=01|stimulus=s001|label=Scrambled|index=1",
    ]


def test_run_time_resolved_decode_passes_metadata_stimulus_anchors(tmp_path: Path, monkeypatch):
    labels = np.tile(np.array([0, 1, 0, 1]), 3)
    groups = np.repeat(["sub-01", "sub-02", "sub-03"], 4)
    stim_files = np.tile(np.array(["stim-a", "stim-b", "stim-a", "stim-b"]), 3)
    times = np.array([0.180, 0.184, 0.188])
    data = np.zeros((len(labels), 2, len(times)), dtype=float)
    for trial_index, label in enumerate(labels):
        data[trial_index, :, :] = label
    metadata = pd.DataFrame({"condition": labels, "group": groups, "stim_file": stim_files})
    epochs = FakeEpochs(data, times, metadata)
    anchor_calls = []

    def fake_align_train_test_features(**kwargs):
        assert kwargs.get("target_labels") is None
        assert kwargs.get("target_anchor_values") is None
        anchor_calls.append(np.asarray(kwargs["train_anchor_values"], dtype=object).copy())
        return SourceAlignmentResult(
            train_features=np.asarray(kwargs["train_features"], dtype=float),
            test_features=np.asarray(kwargs["test_features"], dtype=float),
            metadata={
                "alignment_method": "procrustes",
                "alignment_anchor_mode": "stimulus_id_mean",
                "alignment_anchor_column": "stim_file",
                "alignment_anchor_value_source": "metadata",
                "alignment_target_projection": "group_projection",
                "alignment_n_components": 2,
            },
        )

    monkeypatch.setattr("neureptrace.mne_time_decode.mne.read_epochs", lambda *args, **kwargs: epochs)
    monkeypatch.setattr("neureptrace.mne_time_decode.make_decoder", lambda *args, **kwargs: RecordingFeatureDecoder())
    monkeypatch.setattr("neureptrace.mne_time_decode.align_train_test_features", fake_align_train_test_features)

    results = run_time_resolved_decode(
        epochs_path=tmp_path / "synthetic-epo.fif",
        label_column="condition",
        group_column="group",
        outer_test_groups=("sub-01",),
        out_path=tmp_path / "stimulus_aligned.csv",
        n_splits=3,
        window_ms=1,
        step_ms=4,
        decoder="logistic",
        emission_mode="uncalibrated",
        time_decode_backend="sklearn",
        alignment_method="procrustes",
        alignment_anchor_mode="stimulus_id_mean",
    )

    assert anchor_calls
    np.testing.assert_array_equal(anchor_calls[0], stim_files[groups != "sub-01"])
    assert set(results["alignment_anchor_column"]) == {"stim_file"}


def test_run_time_resolved_decode_passes_target_anchors_for_oracle_stimulus_alignment(tmp_path: Path, monkeypatch):
    labels = np.tile(np.array([0, 1, 0, 1]), 3)
    groups = np.repeat(["sub-01", "sub-02", "sub-03"], 4)
    stim_files = np.tile(np.array(["stim-a", "stim-b", "stim-a", "stim-b"]), 3)
    times = np.array([0.180, 0.184, 0.188])
    data = np.zeros((len(labels), 2, len(times)), dtype=float)
    metadata = pd.DataFrame({"condition": labels, "group": groups, "stim_file": stim_files})
    epochs = FakeEpochs(data, times, metadata)
    target_anchor_calls = []

    def fake_align_train_test_features(**kwargs):
        assert kwargs.get("target_labels") is None
        target_anchor_calls.append(np.asarray(kwargs["target_anchor_values"], dtype=object).copy())
        return SourceAlignmentResult(
            train_features=np.asarray(kwargs["train_features"], dtype=float),
            test_features=np.asarray(kwargs["test_features"], dtype=float),
            metadata={
                "alignment_method": "procrustes",
                "alignment_anchor_mode": "stimulus_id_mean",
                "alignment_anchor_column": "stim_file",
                "alignment_target_projection": "oracle_target_calibrated_alignment",
                "alignment_oracle_target_calibrated": True,
                "alignment_debug_upper_bound": True,
                "alignment_valid_for_benchmark": False,
                "alignment_target_anchor_values_used": True,
                "alignment_protocol_note": "debug upper bound only; not valid for benchmark",
                "alignment_n_components": 2,
            },
        )

    monkeypatch.setattr("neureptrace.mne_time_decode.mne.read_epochs", lambda *args, **kwargs: epochs)
    monkeypatch.setattr("neureptrace.mne_time_decode.make_decoder", lambda *args, **kwargs: RecordingFeatureDecoder())
    monkeypatch.setattr("neureptrace.mne_time_decode.align_train_test_features", fake_align_train_test_features)

    run_time_resolved_decode(
        epochs_path=tmp_path / "synthetic-epo.fif",
        label_column="condition",
        group_column="group",
        outer_test_groups=("sub-01",),
        out_path=tmp_path / "oracle_stimulus_aligned.csv",
        n_splits=3,
        window_ms=1,
        step_ms=4,
        decoder="logistic",
        emission_mode="uncalibrated",
        time_decode_backend="sklearn",
        alignment_method="procrustes",
        alignment_anchor_mode="stimulus_id_mean",
        alignment_target_projection="oracle_target_calibrated_alignment",
    )

    assert target_anchor_calls
    np.testing.assert_array_equal(target_anchor_calls[0], stim_files[groups == "sub-01"])


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


def test_source_time_logit_stacker_writes_bias_and_strong_regularization(tmp_path: Path, monkeypatch):
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
        out_path=tmp_path / "source_time_stacker.csv",
        n_splits=3,
        window_ms=1,
        step_ms=96,
        decoder="logistic",
        emission_mode="uncalibrated",
        source_time_selection="source-oof-logit-stacker",
        source_time_selection_times=(0.088, 0.184),
    )

    assert normalize_source_time_selection("source-oof-logit-stacker") == "source_oof_logit_stacker"
    assert results["decoder"].unique().tolist() == ["logistic_source_oof_logit_stacker"]
    assert results["source_time_selection_weight_type"].unique().tolist() == ["stacker"]
    assert results["source_time_selection_stacker_type"].unique().tolist() == ["shared_time_weights_plus_class_bias"]
    assert results["source_time_selection_stacker_regularization"].unique().tolist() == ["strong"]
    assert results["source_time_selection_class_bias"].str.split("|").map(len).unique().tolist() == [3]
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
