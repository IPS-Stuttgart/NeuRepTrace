"""Alpha-band signal and phase utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.signal

from neureptrace.meg.fieldtrip_struct import get_time_vector, get_trial_signal


def uniform_sample_interval(time_vector) -> float:
    """Return the sample interval after validating a finite regular time axis."""

    time_vector = np.asarray(time_vector, dtype=float).ravel()
    if time_vector.size < 2:
        raise ValueError("time_vector must contain at least two samples.")
    if not np.all(np.isfinite(time_vector)):
        raise ValueError("time_vector must contain only finite values.")

    diffs = np.diff(time_vector)
    if np.any(diffs <= 0):
        raise ValueError("time_vector must be strictly increasing.")

    sample_interval = float(np.median(diffs))
    if not np.allclose(diffs, sample_interval, rtol=1e-6, atol=1e-12):
        raise ValueError("time_vector must be uniformly sampled.")
    return sample_interval


def sampling_rate_from_time_vector(time_vector) -> float:
    """Return sampling rate in Hz after validating ``time_vector``."""

    return float(1.0 / uniform_sample_interval(time_vector))


def _validated_sampling_rate(sampling_rate) -> float:
    try:
        sampling_rate = float(sampling_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError("sampling_rate must be a positive finite value.") from exc
    if not np.isfinite(sampling_rate) or sampling_rate <= 0.0:
        raise ValueError("sampling_rate must be a positive finite value.")
    return sampling_rate


def _validated_signal_values(signal_values) -> np.ndarray:
    signal_values = np.asarray(signal_values, dtype=float)
    if signal_values.ndim == 0:
        raise ValueError("signal_values must have at least one sample dimension.")
    if signal_values.shape[-1] < 2:
        raise ValueError("signal_values must contain at least two samples along the last axis.")
    if not np.all(np.isfinite(signal_values)):
        raise ValueError("signal_values must contain only finite values.")
    return signal_values


def _validated_trial_signal(data, trial_idx: int, time_vector) -> np.ndarray:
    signal = get_trial_signal(data, trial_idx)
    if signal.ndim != 2:
        raise ValueError(f"Trial {trial_idx} must be a 2D channels-by-time array.")
    if signal.shape[1] != np.asarray(time_vector).size:
        raise ValueError(f"Trial {trial_idx} has {signal.shape[1]} samples but its time vector has {np.asarray(time_vector).size} entries.")
    if not np.all(np.isfinite(signal)):
        raise ValueError(f"Trial {trial_idx} signal must contain only finite values.")
    return signal


def _parse_channel_index(value) -> int:
    if not isinstance(value, (int, np.integer)):
        raise ValueError("channel_range must contain integer channel indices.")
    return int(value)


def _channel_indices_from_range(channel_range: Sequence[int], n_channels: int) -> range:
    try:
        start, stop = channel_range
    except (TypeError, ValueError) as exc:
        raise ValueError("channel_range must contain exactly two integer indices.") from exc

    start = _parse_channel_index(start)
    stop = _parse_channel_index(stop)
    if start > stop:
        raise ValueError("channel_range start must be less than or equal to stop.")
    if start < 0 or stop >= int(n_channels):
        raise ValueError(f"channel_range is outside the available channels: got ({start}, {stop}) for {n_channels} channels.")
    return range(start, stop + 1)


def bandpass_filter_signal(signal_values, sampling_rate, lowcut: float = 8.0, highcut: float = 12.0, order: int = 5) -> np.ndarray:
    """Band-pass filter ``signal_values`` along the last axis."""

    signal_values = _validated_signal_values(signal_values)
    sampling_rate = _validated_sampling_rate(sampling_rate)
    nyquist = 0.5 * sampling_rate
    if lowcut <= 0 or highcut <= 0:
        raise ValueError("Cutoff frequencies must be positive.")
    if lowcut >= highcut:
        raise ValueError("lowcut must be lower than highcut.")
    if highcut >= nyquist:
        raise ValueError("highcut must be lower than the Nyquist frequency.")

    sos = scipy.signal.butter(order, [lowcut, highcut], btype="bandpass", fs=sampling_rate, output="sos")
    return scipy.signal.sosfiltfilt(sos, signal_values)


def extract_alpha_signal_and_phase(signal_values, sampling_rate, lowcut: float = 8.0, highcut: float = 12.0) -> tuple[np.ndarray, np.ndarray]:
    """Return alpha-band signal and Hilbert phase."""

    filtered_signal = bandpass_filter_signal(signal_values, sampling_rate, lowcut, highcut)
    analytic_signal = scipy.signal.hilbert(filtered_signal)
    return filtered_signal, np.angle(analytic_signal)


def extract_phase(signal_values, sampling_rate, lowcut: float = 8.0, highcut: float = 12.0) -> np.ndarray:
    """Return the Hilbert phase of the alpha-band-filtered signal."""

    _, phase = extract_alpha_signal_and_phase(signal_values, sampling_rate, lowcut, highcut)
    return phase


def average_phases(phases: Sequence[np.ndarray]) -> np.ndarray:
    """Circularly average phase arrays across channels."""

    if not phases:
        raise ValueError("At least one phase array is required.")
    phase_matrix = np.vstack(phases)
    return np.angle(np.mean(np.exp(1j * phase_matrix), axis=0))


def extract_time_basis(data, trial_idx: int = 0, channel_range: Sequence[int] = (187, 198)) -> np.ndarray:
    """Extract a robust alpha-phase time basis by averaging channel phases."""

    time_vector = get_time_vector(data, trial_idx)
    sampling_rate = sampling_rate_from_time_vector(time_vector)
    signal = _validated_trial_signal(data, trial_idx, time_vector)
    channel_indices = _channel_indices_from_range(channel_range, signal.shape[0])
    phases = [extract_phase(signal[channel_idx, :], sampling_rate) for channel_idx in channel_indices]
    return average_phases(phases)
