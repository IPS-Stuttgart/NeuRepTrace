from pathlib import Path

import numpy as np
import pandas as pd

from neureptrace.mne_time_decode import (
    _apply_epoch_normalization,
    normalize_epoch_normalization,
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


def test_normalize_epoch_normalization_accepts_baseline_whiten_alias():
    assert normalize_epoch_normalization("subject-baseline-whiten") == "subject_baseline_whiten"
    assert normalize_epoch_normalization("identity") == "none"


def test_subject_baseline_whiten_centers_baseline_channel_means():
    rng = np.random.default_rng(31)
    times = np.array([-0.2, -0.1, 0.0, 0.1])
    data = rng.normal(size=(12, 3, 4))
    data[:, :, :2] += np.array([10.0, -3.0, 5.0])[None, :, None]

    normalized = _apply_epoch_normalization(
        data,
        times,
        "subject_baseline_whiten",
        baseline_window=(-0.2, -0.1),
    )

    assert normalized.shape == data.shape
    np.testing.assert_allclose(normalized[:, :, :2].mean(axis=(0, 2)), np.zeros(3), atol=1e-10)


def test_run_time_resolved_decode_supports_multinomial_logistic_with_subject_baseline_whiten(
    tmp_path: Path,
    monkeypatch,
):
    rng = np.random.default_rng(37)
    labels = np.array(["animate", "inanimate"] * 12)
    sessions = np.repeat([f"s{index}" for index in range(6)], 4)
    data = rng.normal(size=(24, 3, 5))
    data[:, :, :2] += np.array([8.0, -4.0, 2.0])[None, :, None]
    data[labels == "animate", 0, 3:] += 1.0
    metadata = pd.DataFrame({"condition": labels, "session": sessions})
    epochs = FakeEpochs(data, np.array([-0.2, -0.1, 0.0, 0.1, 0.2]), metadata)
    monkeypatch.setattr("neureptrace.mne_time_decode.mne.read_epochs", lambda *args, **kwargs: epochs)

    out = tmp_path / "decode_baseline_whiten.csv"
    observations_out = tmp_path / "observations_baseline_whiten.csv"

    results = run_time_resolved_decode(
        epochs_path=tmp_path / "sub-01_epo.fif",
        label_column="condition",
        out_path=out,
        group_column="session",
        n_splits=2,
        window_ms=100,
        step_ms=100,
        max_iter=2000,
        decoder="multinomial-logistic",
        emission_mode="calibrated",
        normalization="subject_baseline_whiten",
        baseline_window=(-0.2, -0.1),
        observation_out_path=observations_out,
    )
    observations = pd.read_csv(observations_out)

    assert results["decoder"].unique().tolist() == ["multinomial-logistic"]
    assert results["normalization"].unique().tolist() == ["subject_baseline_whiten"]
    assert observations["decoder"].unique().tolist() == ["multinomial-logistic"]
    assert observations["normalization"].unique().tolist() == ["subject_baseline_whiten"]
    assert observations[["prob_class_0", "prob_class_1"]].sum(axis=1).round(6).tolist() == [1.0] * len(observations)
