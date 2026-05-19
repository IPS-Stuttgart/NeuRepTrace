from __future__ import annotations

import numpy as np

from neureptrace.meg.alpha_metrics import AlphaMetricConfig, compute_alpha_metrics
from neureptrace.meg.alpha_movement import AlphaMovementConfig, compute_alpha_movement, sample_time_indices, summarize_alpha_movement
from neureptrace.meg.alpha_signal import uniform_sample_interval
from neureptrace.meg.sensor_geometry import select_channels


def _synthetic_fieldtrip_data(n_trials: int = 4, n_channels: int = 5, n_times: int = 121):
    sfreq = 100.0
    times = np.arange(n_times) / sfreq - 0.6
    phase_offsets = np.linspace(0.0, np.pi / 2, n_channels)
    trials = []
    for trial in range(n_trials):
        signal = np.vstack([np.sin(2 * np.pi * 10.0 * times + phase_offsets[channel] + trial * 0.05) for channel in range(n_channels)])
        trials.append(signal)

    trial_cell = np.empty((1, n_trials), dtype=object)
    time_cell = np.empty((1, n_trials), dtype=object)
    for trial, signal in enumerate(trials):
        trial_cell[0, trial] = signal
        time_cell[0, trial] = times[None, :]

    label = np.array([["MLO001"], ["MLO002"], ["MRO001"], ["MZO001"], ["MLT001"], ["STATUS"]], dtype=object)
    angles = np.linspace(0.0, 2 * np.pi, n_channels, endpoint=False)
    chanpos = np.column_stack([50.0 * np.cos(angles), 50.0 * np.sin(angles), np.zeros(n_channels)])
    grad = {"chanpos": chanpos, "unit": "mm"}
    trialinfo = np.array([[1], [2], [1], [2]])
    return {"label": label, "trial": trial_cell, "time": time_cell, "trialinfo": trialinfo, "grad": grad}


def test_uniform_sample_interval_validates_regular_axis():
    assert uniform_sample_interval([0.0, 0.1, 0.2]) == 0.1


def test_select_channels_trims_to_trial_channel_count():
    data = _synthetic_fieldtrip_data()
    assert select_channels(data, r"^M[LRZ]O") == [0, 1, 2, 3]


def test_compute_alpha_metrics_returns_trial_rows():
    data = _synthetic_fieldtrip_data()
    rows = compute_alpha_metrics(data, participant_id=10, config=AlphaMetricConfig(time_window=(-0.4, -0.05)))
    assert len(rows) == 4
    assert rows[0]["participant"] == 10
    assert rows[0]["n_channels"] == 4
    assert rows[0]["alpha_power"] > 0.0
    assert "phase_plane_fit" in rows[0]


def test_alpha_movement_and_summary():
    data = _synthetic_fieldtrip_data()
    config = AlphaMovementConfig(time_window=(-0.4, 0.0), trajectory_step_s=0.1)
    rows = compute_alpha_movement(data, participant_id=10, config=config)
    assert rows
    assert {row["trial"] for row in rows} == {0, 1, 2, 3}
    assert rows[0]["n_channels"] == 5
    summary = summarize_alpha_movement(rows)
    assert summary
    assert "mean_trajectory_projected_displacement_mm" in summary[0]


def test_sample_time_indices_accepts_none_step():
    times = np.linspace(-0.5, 0.5, 11)
    indices = sample_time_indices(times, (-0.2, 0.2), None)
    assert indices.tolist() == [3, 4, 5, 6, 7]
