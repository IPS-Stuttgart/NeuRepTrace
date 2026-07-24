from __future__ import annotations

import numpy as np
import pytest

from neureptrace.signal import validate_time_axis as exported_validate_time_axis
from neureptrace.signal.band import (
    average_phases,
    bandpass_filter,
    circular_mean_phase,
    validate_band_hz,
    validate_sampling_rate,
    validate_signal_values,
    validate_time_axis,
)


def test_time_axis_rejects_complex_arrays_before_float_coercion() -> None:
    time_vector = np.array([0.0 + 1.0j, 0.01 + 1.0j, 0.02 + 1.0j])

    with pytest.raises(ValueError, match="real-valued time values"):
        validate_time_axis(time_vector)

    with pytest.raises(ValueError, match="real-valued time values"):
        exported_validate_time_axis(value for value in time_vector)


def test_signal_values_reject_complex_samples_before_filtering() -> None:
    sampling_rate = 200.0
    time = np.arange(400, dtype=float) / sampling_rate
    signal = np.sin(2.0 * np.pi * 10.0 * time).astype(complex)
    signal += 0.25j

    with pytest.raises(ValueError, match="real-valued samples"):
        validate_signal_values(signal)

    with pytest.raises(ValueError, match="real-valued samples"):
        bandpass_filter(signal, sampling_rate, (8.0, 12.0))


def test_sampling_rate_and_cutoffs_reject_numpy_complex_scalars() -> None:
    with pytest.raises(ValueError, match="not complex"):
        validate_sampling_rate(np.complex128(200.0 + 1.0j))

    with pytest.raises(ValueError, match="real-valued"):
        validate_band_hz((np.complex128(8.0 + 1.0j), 12.0), 200.0)

    with pytest.raises(ValueError, match="not complex"):
        validate_band_hz((8.0, 12.0), np.complex128(200.0 + 1.0j))


def test_phase_helpers_reject_complex_values() -> None:
    phases = np.array([0.0 + 1.0j, np.pi / 2.0 + 1.0j])

    with pytest.raises(ValueError, match="real-valued phase values"):
        circular_mean_phase(phases)

    with pytest.raises(ValueError, match="real-valued phase values"):
        average_phases([phases, phases])
