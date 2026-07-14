from __future__ import annotations

import numpy as np
import pytest
from neureptrace.signal.band import (
    average_phases,
    band_analytic_signal,
    bandpass_filter,
    circular_mean_phase,
    extract_alpha_signal_and_phase,
    sampling_rate_from_time_axis,
    validate_time_axis,
)


def test_sampling_rate_from_time_axis_validates_uniform_axis() -> None:
    assert sampling_rate_from_time_axis([0.0, 0.01, 0.02]) == pytest.approx(100.0)

    with pytest.raises(ValueError, match="uniformly sampled"):
        sampling_rate_from_time_axis([0.0, 0.01, 0.03])


def test_validate_time_axis_rejects_nonuniform_tiny_intervals() -> None:
    np.testing.assert_allclose(validate_time_axis([0.0, 1e-15, 2e-15]), [0.0, 1e-15, 2e-15])

    with pytest.raises(ValueError, match="uniformly sampled"):
        validate_time_axis([0.0, 1e-15, 4e-15])


def test_validate_time_axis_accepts_one_pass_iterables() -> None:
    axis = validate_time_axis(value for value in (0.0, 0.01, 0.02))

    np.testing.assert_allclose(axis, [0.0, 0.01, 0.02])


def test_validate_time_axis_rejects_multidimensional_axes() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        validate_time_axis(np.array([[0.0, 0.01], [0.02, 0.03]]))


def test_validate_time_axis_rejects_boolean_values() -> None:
    with pytest.raises(ValueError, match="not boolean"):
        validate_time_axis([False, True])

    with pytest.raises(ValueError, match="not boolean"):
        validate_time_axis([0.0, True])

    with pytest.raises(ValueError, match="not boolean"):
        validate_time_axis(value for value in (0.0, True))


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


def test_bandpass_filter_accepts_nested_one_pass_signal_rows() -> None:
    sampling_rate = 200.0
    time = np.arange(400, dtype=float) / sampling_rate
    signal = np.vstack(
        [
            np.sin(2 * np.pi * 10.0 * time),
            np.sin(2 * np.pi * 10.0 * time + 0.2),
        ]
    )
    one_pass_signal = ((float(value) for value in row) for row in signal)

    filtered = bandpass_filter(one_pass_signal, sampling_rate, (8.0, 12.0))

    assert filtered.shape == signal.shape
    assert np.all(np.isfinite(filtered))


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


def test_phase_helpers_accept_one_pass_iterables() -> None:
    mean_phase = circular_mean_phase(value for value in (0.0, np.pi / 2.0))
    phase_rows = ((value for value in row) for row in ([0.0, np.pi], [0.0, np.pi]))

    assert mean_phase == pytest.approx(np.pi / 4.0)
    np.testing.assert_allclose(average_phases(phase_rows), [0.0, np.pi])


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
