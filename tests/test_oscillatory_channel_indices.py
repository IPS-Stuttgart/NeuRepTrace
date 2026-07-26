from __future__ import annotations

import numpy as np
import pytest

from neureptrace.features.oscillatory import (
    compute_band_features,
    compute_band_trial_features,
)


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    sampling_rate = 200.0
    time = np.arange(400, dtype=float) / sampling_rate - 0.5
    data = np.stack(
        [
            np.sin(2.0 * np.pi * 10.0 * time),
            np.cos(2.0 * np.pi * 10.0 * time),
            np.sin(2.0 * np.pi * 10.0 * time + 0.25),
        ]
    )
    return data, time


@pytest.mark.parametrize(
    "channel_indices",
    [
        [True],
        [0.5],
        np.asarray([[0, 1]], dtype=int),
        [],
        1,
    ],
)
def test_band_trial_features_rejects_lossy_channel_indices(channel_indices: object) -> None:
    signal, time = _fixture()

    with pytest.raises(
        ValueError,
        match="channel_indices must be a non-empty one-dimensional sequence of integers",
    ):
        compute_band_trial_features(
            signal,
            time,
            time_window=(-0.25, 0.25),
            channel_indices=channel_indices,  # type: ignore[arg-type]
            outputs=("mean_power",),
        )


def test_band_features_rejects_fractional_channel_indices_before_truncation() -> None:
    signal, time = _fixture()

    with pytest.raises(ValueError, match="sequence of integers"):
        compute_band_features(
            signal[np.newaxis, ...],
            time,
            windows=(-0.25, 0.25),
            channel_indices=[1.9],
            outputs=("mean_power",),
        )


def test_band_features_accepts_numpy_integer_channel_indices() -> None:
    signal, time = _fixture()

    rows = compute_band_features(
        signal[np.newaxis, ...],
        time,
        windows=(-0.25, 0.25),
        channel_indices=np.asarray([0, 2], dtype=np.int64),
        outputs=("mean_power",),
    )

    assert len(rows) == 1
    assert rows[0]["n_channels"] == 2
    assert rows[0]["mean_power"] > 0.0
