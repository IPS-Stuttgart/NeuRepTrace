from __future__ import annotations

import numpy as np
import pytest

from neureptrace.signal.band import (
    average_phases,
    band_analytic_signal,
    bandpass_filter,
    extract_alpha_signal_and_phase,
    sampling_rate_from_time_axis,
)


def test_sampling_rate_from_time_axis_validates_uniform_axis() -> None:
    assert sampling_rate_from_time_axis([0.0, 0.01, 0.02]) == pytest.approx(100.0)

    with pytest.raises(ValueError, match="uniformly sampled"):
        sampling_rate_from_time_axis([0.0, 0.01, 0.03])


def test_bandpass_filter_and_hilbert_keep_shape() -> None:
    sampling_rate = 200.0
    time = np.arange(400, dtype=float) / sampling_rate
    signal = np.vstack(
        [
            np.sin(2 * np.pi * 10.0 * time),
            np.sin(2 * np.pi * 10.0 * time + 0.2),
        ]
    )

    filtered = bandpass_filter(signal, sampling_rate, (8.0, 12.0))
    analytic = band_analytic_signal(signal, sampling_rate, (8.0, 12.0))
    _alpha, phase = extract_alpha_signal_and_phase(signal, sampling_rate)

    assert filtered.shape == signal.shape
    assert analytic.shape == signal.shape
    assert phase.shape == signal.shape
    assert np.iscomplexobj(analytic)


def test_bandpass_filter_rejects_too_short_signal_with_clear_error() -> None:
    sampling_rate = 200.0
    time = np.arange(10, dtype=float) / sampling_rate
    signal = np.sin(2 * np.pi * 10.0 * time)

    with pytest.raises(ValueError, match="more than 33 samples along axis 0"):
        bandpass_filter(signal, sampling_rate, (8.0, 12.0))


def test_average_phases_matches_circular_mean() -> None:
    phases = [np.array([0.0, np.pi]), np.array([0.0, np.pi])]

    np.testing.assert_allclose(average_phases(phases), [0.0, np.pi])

    with pytest.raises(ValueError, match="At least one"):
        average_phases([])


def test_average_phases_preserves_multidimensional_shape() -> None:
    phases = [
        np.array([[0.0, np.pi], [np.pi / 2.0, -np.pi / 2.0]]),
        np.array([[0.0, np.pi], [np.pi / 2.0, -np.pi / 2.0]]),
    ]

    averaged = average_phases(phases)

    assert averaged.shape == phases[0].shape
    np.testing.assert_allclose(averaged, phases[0])


def test_average_phases_rejects_mismatched_shapes() -> None:
    phases = [np.zeros((2, 2)), np.zeros((4,))]

    with pytest.raises(ValueError, match="same shape"):
        average_phases(phases)
