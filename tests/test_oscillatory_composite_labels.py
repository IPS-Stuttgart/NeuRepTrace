from __future__ import annotations

import numpy as np

from neureptrace.features.oscillatory import compute_band_features


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    sampling_rate = 200.0
    time = np.arange(400, dtype=float) / sampling_rate - 0.5
    data = np.zeros((2, 3, time.size), dtype=float)
    for trial in range(data.shape[0]):
        for channel in range(data.shape[1]):
            data[trial, channel] = np.sin(2 * np.pi * 10.0 * time + 0.1 * trial + 0.2 * channel)
    return data, time


def test_compute_band_features_preserves_tuple_trial_labels() -> None:
    data, time = _fixture()
    labels = [("face", "left"), ("house", "right")]

    rows = compute_band_features(
        data,
        time,
        windows=[(-0.25, 0.25)],
        labels=labels,
        outputs=("mean_power",),
    )

    assert [row["label"] for row in rows] == labels
    assert all("mean_power" in row for row in rows)


def test_compute_band_features_accepts_single_column_label_arrays() -> None:
    data, time = _fixture()
    labels = np.asarray([["face"], ["house"]], dtype=object)

    rows = compute_band_features(
        data,
        time,
        windows=[(-0.25, 0.25)],
        labels=labels,
        outputs=("mean_power",),
    )

    assert [row["label"] for row in rows] == ["face", "house"]
