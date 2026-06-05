from pathlib import Path

import numpy as np
import pandas as pd

from neureptrace.mne_time_decode_foldlocal import _normalize_epoch_data_for_fold, run_time_resolved_decode


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


def test_subject_z_normalization_uses_train_fold_statistics_only():
    data = np.array(
        [
            [[0.0, 0.0, 0.0]],
            [[2.0, 2.0, 2.0]],
            [[100.0, 100.0, 100.0]],
        ]
    )
    times = np.array([-0.1, 0.0, 0.1])

    normalized = _normalize_epoch_data_for_fold(
        data,
        times,
        "subject_z",
        baseline_window=(-0.1, 0.0),
        train_idx=np.array([0, 1]),
    )

    np.testing.assert_allclose(normalized[:2].mean(axis=(0, 2), keepdims=True), 0.0)
    np.testing.assert_allclose(normalized[:2].std(axis=(0, 2), keepdims=True), 1.0)
    np.testing.assert_allclose(normalized[2], 99.0)


def test_subject_baseline_z_normalization_uses_train_baseline_only():
    data = np.array(
        [
            [[0.0, 0.0, 10.0]],
            [[2.0, 2.0, 20.0]],
            [[100.0, 100.0, 30.0]],
        ]
    )
    times = np.array([-0.2, -0.1, 0.1])

    normalized = _normalize_epoch_data_for_fold(
        data,
        times,
        "subject_baseline_z",
        baseline_window=(-0.2, -0.1),
        train_idx=np.array([0, 1]),
    )

    np.testing.assert_allclose(normalized[:2, :, :2].mean(axis=(0, 2), keepdims=True), 0.0)
    np.testing.assert_allclose(normalized[:2, :, :2].std(axis=(0, 2), keepdims=True), 1.0)
    np.testing.assert_allclose(normalized[2, :, :2], [[99.0, 99.0]])


def test_subject_baseline_whiten_fit_excludes_test_fold_outlier():
    data = np.array(
        [
            [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
            [[2.0, 2.0, 3.0], [0.0, 0.0, 4.0]],
            [[100.0, 100.0, 5.0], [100.0, 100.0, 6.0]],
        ]
    )
    times = np.array([-0.2, -0.1, 0.1])

    fold_local = _normalize_epoch_data_for_fold(
        data,
        times,
        "subject_baseline_whiten",
        baseline_window=(-0.2, -0.1),
        train_idx=np.array([0, 1]),
    )

    contaminated = _normalize_epoch_data_for_fold(
        data,
        times,
        "subject_baseline_whiten",
        baseline_window=(-0.2, -0.1),
        train_idx=np.array([0, 1, 2]),
    )

    assert abs(float(fold_local[2, 0, 0])) > abs(float(contaminated[2, 0, 0]))


def test_foldlocal_source_time_selection_is_inner_source_only(tmp_path: Path, monkeypatch):
    rng = np.random.default_rng(31)
    subjects = np.repeat(["sub-01", "sub-02", "sub-03"], 6)
    labels = np.tile(["left", "right"], 9)
    times = np.array([0.00, 0.01, 0.02, 0.03, 0.04, 0.05])
    data = rng.normal(scale=0.05, size=(len(labels), 1, len(times)))
    data[labels == "left", 0, 2:4] += 0.8
    data[labels == "right", 0, 2:4] -= 0.8
    metadata = pd.DataFrame({"condition": labels, "subject": subjects})
    epochs = FakeEpochs(data, times, metadata)
    monkeypatch.setattr("neureptrace.mne_time_decode.mne.read_epochs", lambda *args, **kwargs: epochs)

    out = tmp_path / "foldlocal_source_time.csv"
    observations_out = tmp_path / "foldlocal_source_time_observations.csv"
    results = run_time_resolved_decode(
        epochs_path=tmp_path / "epochs.fif",
        label_column="condition",
        group_column="subject",
        out_path=out,
        n_splits=3,
        window_ms=20,
        step_ms=20,
        max_iter=1000,
        emission_mode="uncalibrated",
        normalization="subject_baseline_z",
        baseline_window=(0.00, 0.01),
        source_time_selection="source_oof_time_weighted_logits",
        source_time_selection_times=(0.005, 0.025, 0.045),
        source_time_selection_output_time=0.025,
        observation_out_path=observations_out,
    )
    observations = pd.read_csv(observations_out)

    assert results["temporal_mode"].unique().tolist() == ["source_oof_time_weighted_logits"]
    assert results["time"].unique().tolist() == [0.025]
    assert results["source_time_selection_normalization_scope"].unique().tolist() == ["inner_train_fold"]
    assert results["source_time_selection_weights"].str.split("|").map(len).unique().tolist() == [3]
    assert observations["temporal_mode"].unique().tolist() == ["source_oof_time_weighted_logits"]
    assert observations["source_time_selection_normalization_scope"].unique().tolist() == ["inner_train_fold"]
    assert observations[["prob_class_0", "prob_class_1"]].sum(axis=1).round(6).tolist() == [1.0] * len(observations)
