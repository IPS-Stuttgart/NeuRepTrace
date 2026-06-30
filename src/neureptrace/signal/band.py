"""Generic band-limited signal-processing primitives.

The functions in this module deliberately avoid assumptions about a particular
MEG/EEG file format.  They operate on NumPy-like arrays and treat the last axis
as time by default, which makes them reusable by project-specific loaders and
feature extractors.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.signal

BandHz = tuple[float, float]


def _is_bool_like(value: object) -> bool:
    return isinstance(value, (bool, np.bool_))


def _contains_bool_like(value: object) -> bool:
    if _is_bool_like(value):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.bool_):
            return True
        if value.dtype == object:
            return any(_contains_bool_like(item) for item in value.ravel())
        return False
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, Sequence):
        return any(_contains_bool_like(item) for item in value)
    return False


def _normalize_axis(axis: int, ndim: int) -> int:
    if _is_bool_like(axis):
        raise ValueError("axis must be an integer, not boolean.")
    if not isinstance(axis, (int, np.integer)):
        raise ValueError("axis must be an integer.")
    axis = int(axis)
    if ndim <= 0:
        raise ValueError("array must have at least one dimension.")
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"axis {axis} is out of bounds for an array with {ndim} dimensions.")
    return axis


def validate_time_axis(time_vector) -> np.ndarray:
    """Return a validated, one-dimensional, uniformly sampled time axis."""

    if _contains_bool_like(time_vector):
        raise ValueError("time_vector must contain numeric time values, not boolean values.")
    time_vector = np.asarray(time_vector, dtype=float)
    if time_vector.ndim == 0:
        raise ValueError("time_vector must contain at least two samples.")
    if time_vector.ndim != 1:
        non_singleton_axes = sum(size > 1 for size in time_vector.shape)
        if non_singleton_axes != 1:
            raise ValueError("time_vector must be one-dimensional.")
        time_vector = time_vector.reshape(-1)
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
    return time_vector


def uniform_sample_interval(time_vector) -> float:
    """Return the sample interval in seconds after validating ``time_vector``."""

    time_vector = validate_time_axis(time_vector)
    return float(np.median(np.diff(time_vector)))


def sampling_rate_from_time_axis(time_vector) -> float:
    """Return sampling rate in Hz after validating ``time_vector``."""

    return float(1.0 / uniform_sample_interval(time_vector))


# Compatibility alias for older PyMEGDec naming.
sampling_rate_from_time_vector = sampling_rate_from_time_axis


def validate_sampling_rate(sampling_rate) -> float:
    """Return a positive finite sampling rate in Hz."""

    if _is_bool_like(sampling_rate):
        raise ValueError("sampling_rate must be a positive finite value, not boolean.")
    try:
        sampling_rate = float(sampling_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError("sampling_rate must be a positive finite value.") from exc
    if not np.isfinite(sampling_rate) or sampling_rate <= 0.0:
        raise ValueError("sampling_rate must be a positive finite value.")
    return sampling_rate


def validate_signal_values(signal_values, *, axis: int = -1) -> np.ndarray:
    """Return finite real-valued signal samples with at least two time samples."""

    signal_values = np.asarray(signal_values, dtype=float)
    if signal_values.ndim == 0:
        raise ValueError("signal_values must have at least one sample dimension.")
    axis = _normalize_axis(axis, signal_values.ndim)
    if signal_values.shape[axis] < 2:
        raise ValueError(f"signal_values must contain at least two samples along axis {axis}.")
    if not np.all(np.isfinite(signal_values)):
        raise ValueError("signal_values must contain only finite values.")
    return signal_values


def validate_band_hz(band_hz: Sequence[float], sampling_rate) -> BandHz:
    """Return validated ``(low, high)`` band edges in Hz."""

    sampling_rate = validate_sampling_rate(sampling_rate)
    try:
        lowcut, highcut = band_hz
    except (TypeError, ValueError) as exc:
        raise ValueError("band_hz must contain exactly two cutoff frequencies.") from exc

    if _is_bool_like(lowcut) or _is_bool_like(highcut):
        raise ValueError("Cutoff frequencies must be finite numbers, not boolean.")
    lowcut = float(lowcut)
    highcut = float(highcut)
    if not np.isfinite(lowcut) or not np.isfinite(highcut):
        raise ValueError("Cutoff frequencies must be finite.")
    if lowcut <= 0 or highcut <= 0:
        raise ValueError("Cutoff frequencies must be positive.")
    if lowcut >= highcut:
        raise ValueError("lowcut must be lower than highcut.")
    nyquist = 0.5 * sampling_rate
    if highcut >= nyquist:
        raise ValueError("highcut must be lower than the Nyquist frequency.")
    return lowcut, highcut


def _validate_filter_order(order) -> int:
    if _is_bool_like(order):
        raise ValueError("filter order must be a positive integer, not boolean.")
    if not isinstance(order, (int, np.integer)):
        raise ValueError("filter order must be a positive integer.")
    order = int(order)
    if order <= 0:
        raise ValueError("filter order must be a positive integer.")
    return order


def bandpass_sos(sampling_rate, band_hz: Sequence[float] = (8.0, 12.0), *, order: int = 5) -> np.ndarray:
    """Design a Butterworth band-pass filter as second-order sections."""

    sampling_rate = validate_sampling_rate(sampling_rate)
    lowcut, highcut = validate_band_hz(band_hz, sampling_rate)
    order = _validate_filter_order(order)
    return scipy.signal.butter(
        order,
        [lowcut, highcut],
        btype="bandpass",
        fs=sampling_rate,
        output="sos",
    )


def _default_sosfiltfilt_padlen(sos: np.ndarray) -> int:
    """Return SciPy's default ``sosfiltfilt`` padding length for an SOS filter."""

    return int(3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum())))


def _validate_sosfiltfilt_length(signal_values: np.ndarray, *, axis: int, sos: np.ndarray) -> None:
    padlen = _default_sosfiltfilt_padlen(sos)
    n_samples = signal_values.shape[axis]
    if n_samples <= padlen:
        raise ValueError(
            f"signal_values must contain more than {padlen} samples along axis {axis} for zero-phase filtering; got {n_samples}."
        )


def bandpass_filter(signal_values, sampling_rate, band_hz: Sequence[float] = (8.0, 12.0), *, order: int = 5, axis: int = -1) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter along ``axis``."""

    signal_values = validate_signal_values(signal_values, axis=axis)
    axis = _normalize_axis(axis, signal_values.ndim)
    sos = bandpass_sos(sampling_rate, band_hz, order=order)
    _validate_sosfiltfilt_length(signal_values, axis=axis, sos=sos)
    return scipy.signal.sosfiltfilt(sos, signal_values, axis=axis)


def bandpass_filter_signal(signal_values, sampling_rate, lowcut: float = 8.0, highcut: float = 12.0, order: int = 5, *, axis: int = -1) -> np.ndarray:
    """Compatibility wrapper around :func:`bandpass_filter` using low/high args."""

    return bandpass_filter(
        signal_values,
        sampling_rate,
        (lowcut, highcut),
        order=order,
        axis=axis,
    )


def band_analytic_signal(signal_values, sampling_rate, band_hz: Sequence[float] = (8.0, 12.0), *, order: int = 5, axis: int = -1) -> np.ndarray:
    """Return the complex Hilbert analytic signal after band-pass filtering."""

    filtered_signal = bandpass_filter(signal_values, sampling_rate, band_hz, order=order, axis=axis)
    return scipy.signal.hilbert(filtered_signal, axis=axis)


def extract_band_signal_and_phase(signal_values, sampling_rate, band_hz: Sequence[float] = (8.0, 12.0), *, order: int = 5, axis: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """Return band-pass filtered real signal and Hilbert phase."""

    filtered_signal = bandpass_filter(signal_values, sampling_rate, band_hz, order=order, axis=axis)
    analytic_signal = scipy.signal.hilbert(filtered_signal, axis=axis)
    return filtered_signal, np.angle(analytic_signal)


def extract_alpha_signal_and_phase(signal_values, sampling_rate, lowcut: float = 8.0, highcut: float = 12.0, order: int = 5, *, axis: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """Return alpha-band filtered real signal and Hilbert phase."""

    return extract_band_signal_and_phase(
        signal_values,
        sampling_rate,
        (lowcut, highcut),
        order=order,
        axis=axis,
    )


def extract_phase(signal_values, sampling_rate, lowcut: float = 8.0, highcut: float = 12.0, order: int = 5, *, axis: int = -1) -> np.ndarray:
    """Return Hilbert phase after band-pass filtering."""

    _, phase = extract_alpha_signal_and_phase(
        signal_values,
        sampling_rate,
        lowcut,
        highcut,
        order=order,
        axis=axis,
    )
    return phase


def circular_mean_phase(phases, *, axis=None) -> np.ndarray:
    """Return circular mean phase for phase values in radians."""

    phase_array = np.asarray(phases, dtype=float)
    if phase_array.size == 0:
        raise ValueError("At least one phase value is required.")
    if not np.all(np.isfinite(phase_array)):
        raise ValueError("phases must contain only finite values.")
    return np.angle(np.mean(np.exp(1j * phase_array), axis=axis))


def average_phases(phases) -> np.ndarray:
    """Average a non-empty collection of equally shaped phase arrays."""

    phase_list = [np.asarray(phase, dtype=float) for phase in phases]
    if not phase_list:
        raise ValueError("At least one phase array is required.")

    reference_shape = phase_list[0].shape
    if any(phase.shape != reference_shape for phase in phase_list):
        raise ValueError("All phase arrays must have the same shape.")

    phase_stack = np.stack(phase_list, axis=0)
    return circular_mean_phase(phase_stack, axis=0)
