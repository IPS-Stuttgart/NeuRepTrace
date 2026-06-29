from __future__ import annotations

import numpy as np
import pytest

from neureptrace.features.oscillatory import (
    BandFeatureWindow,
    compute_alpha_features,
    compute_band_analytic_window,
    compute_band_features,
    compute_band_trial_features,
    summarize_analytic_window,
)


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    sampling_rate = 200.0
    time = np.arange(400, dtype=float) / sampling_rate - 0.5
    data = np.zeros((2, 3, time.size), dtype=float)
    for trial in range(data.shape[0]):
        for channel in range(data.shape[1]):
            data[trial, channel] = np.sin(2 * np.pi * 10.0 * time + 0.1 * trial + 0.2 * channel)
    return data, time


@pytest.mark.parametrize("axis", [False, True, np.bool_(True)])
def test_compute_band_features_rejects_boolean_axes(axis) -> None:
    data, time = _fixture()

    with pytest.raises(ValueError, match="axis must be an integer"):
        compute_band_features(data, time, trial_axis=axis, windows=[(-0.25, 0.25)])


def test_compute_band_features_accepts_numpy_integer_axes() -> None:
    data, time = _fixture()

    rows = compute_band_features(
        data,
        time,
        trial_axis=np.int64(0),
        channel_axis=np.int64(1),
        time_axis=np.int64(-1),
        windows=[(-0.25, 0.25)],
    )

    assert len(rows) == data.shape[0]


@pytest.mark.parametrize(
    "window",
    [
        (False, 0.25),
        (-0.25, True),
        ("named", False, 0.25),
        {"name": "mapped", "start": -0.25, "stop": np.bool_(True)},
        BandFeatureWindow("dataclass", -0.25, True),
    ],
)
def test_compute_band_features_rejects_boolean_window_endpoints(window) -> None:
    data, time = _fixture()

    with pytest.raises(ValueError, match="window .* must be"):
        compute_band_features(data, time, windows=[window])


@pytest.mark.parametrize(
    "window",
    [
        (np.asarray(-0.25), 0.25),
        (-0.25, np.asarray([0.25])),
        ("named", np.asarray([-0.25]), 0.25),
        {"name": "mapped", "start": -0.25, "stop": np.asarray([[0.25]])},
        BandFeatureWindow("dataclass", np.asarray(-0.25), 0.25),
    ],
)
def test_compute_band_features_rejects_array_window_endpoints(window) -> None:
    data, time = _fixture()

    with pytest.raises(ValueError, match="window .* must be"):
        compute_band_features(data, time, windows=[window])


def test_compute_band_analytic_window_accepts_matlab_row_time_axis() -> None:
    data, time = _fixture()

    alpha_window, indices = compute_band_analytic_window(
        data[0],
        time[None, :],
        band_hz=(8.0, 12.0),
        time_window=(-0.25, 0.25),
    )

    assert alpha_window.shape == (3, indices.size)
    assert np.iscomplexobj(alpha_window)


def test_summarize_analytic_window_returns_requested_features() -> None:
    data, time = _fixture()
    alpha_window, _ = compute_band_analytic_window(data[0], time, time_window=(-0.25, 0.25))

    summary = summarize_analytic_window(alpha_window)

    assert summary["mean_power"] > 0.0
    assert summary["log_power"] < np.log(summary["mean_power"] + 1.0)
    assert 0.0 <= summary["phase_concentration"] <= 1.0
    assert -np.pi <= summary["mean_phase"] <= np.pi


def test_compute_band_trial_features_respects_channel_indices() -> None:
    data, time = _fixture()

    features = compute_band_trial_features(
        data[0],
        time,
        time_window=(-0.25, 0.25),
        channel_indices=[0, 2],
        outputs=("mean_power", "phase_concentration"),
    )

    assert set(features) == {"mean_power", "phase_concentration"}
    assert features["mean_power"] > 0.0


def test_compute_band_features_returns_one_row_per_trial_and_window() -> None:
    data, time = _fixture()

    rows = compute_band_features(
        data,
        time,
        windows=[BandFeatureWindow("prestim", -0.4, -0.1), ("poststim", 0.1, 0.4)],
        channel_indices=[0, 1],
        labels=["a", "b"],
    )

    assert len(rows) == 4
    assert rows[0]["window"] == "prestim"
    assert rows[0]["label"] == "a"
    assert rows[0]["n_channels"] == 2
    assert "mean_power" in rows[0]


def test_compute_alpha_features_uses_alpha_default_band() -> None:
    data, time = _fixture()

    rows = compute_alpha_features(data, time, windows=[(-0.25, 0.25)])

    assert rows[0]["low_freq"] == pytest.approx(8.0)
    assert rows[0]["high_freq"] == pytest.approx(12.0)
