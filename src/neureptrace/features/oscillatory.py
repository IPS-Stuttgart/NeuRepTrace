"""Reusable oscillatory band feature extraction.

This module builds on :mod:`neureptrace.signal.band` and intentionally operates
on arrays rather than FieldTrip, MNE, or project-specific data structures.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from neureptrace.signal.band import (
    band_analytic_signal,
    circular_mean_phase,
    sampling_rate_from_time_axis,
    validate_band_hz,
    validate_signal_values,
    validate_time_axis,
)

DEFAULT_BAND_FEATURE_OUTPUTS = (
    "mean_power",
    "log_power",
    "phase_concentration",
    "mean_phase",
)


def _is_bool_like(value: object) -> bool:
    return isinstance(value, (bool, np.bool_))


@dataclass(frozen=True)
class BandFeatureWindow:
    """Named time window in seconds for oscillatory feature extraction."""

    name: str
    start: float
    stop: float

    @property
    def as_tuple(self) -> tuple[float, float]:
        return (self.start, self.stop)


def _normalize_axis(axis: int, ndim: int) -> int:
    if _is_bool_like(axis) or not isinstance(axis, (int, np.integer)):
        raise ValueError("axis must be an integer.")
    axis = int(axis)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"axis {axis} is out of bounds for an array with {ndim} dimensions.")
    return axis


def _validate_signal_time_axis(signal, time_vector, *, time_axis: int = -1) -> tuple[np.ndarray, np.ndarray, float, int]:
    signal = validate_signal_values(signal, axis=time_axis)
    time_axis = _normalize_axis(time_axis, signal.ndim)
    time_vector = validate_time_axis(time_vector)
    if signal.shape[time_axis] != time_vector.size:
        axis_description = "last axis" if time_axis == signal.ndim - 1 else f"axis {time_axis}"
        raise ValueError(
            f"signal has {signal.shape[time_axis]} samples along its {axis_description} but time_vector has "
            f"{time_vector.size} entries."
        )
    return signal, time_vector, sampling_rate_from_time_axis(time_vector), time_axis


def _window_endpoint_message(name: str, *, allow_negative_infinity: bool = False, allow_positive_infinity: bool = False) -> str:
    if allow_negative_infinity:
        expected = "finite or -inf"
    elif allow_positive_infinity:
        expected = "finite or inf"
    else:
        expected = "finite"
    return f"window {name} must be {expected}."


def _normalize_window_endpoint(
    value: object,
    *,
    name: str,
    allow_negative_infinity: bool = False,
    allow_positive_infinity: bool = False,
) -> float:
    message = _window_endpoint_message(
        name,
        allow_negative_infinity=allow_negative_infinity,
        allow_positive_infinity=allow_positive_infinity,
    )
    if _is_bool_like(value):
        raise ValueError(message)
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if np.isfinite(numeric):
        return numeric
    if allow_negative_infinity and np.isneginf(numeric):
        return numeric
    if allow_positive_infinity and np.isposinf(numeric):
        return numeric
    raise ValueError(message)


def _normalize_window(window, *, default_name: str = "window") -> BandFeatureWindow:
    if isinstance(window, BandFeatureWindow):
        name = str(window.name)
        start = window.start
        stop = window.stop
    elif isinstance(window, Mapping):
        name = str(window.get("name", default_name))
        try:
            start = window["start"]
            stop = window["stop"]
        except KeyError as exc:
            raise ValueError("window mappings must contain 'start' and 'stop'.") from exc
    else:
        values = tuple(window)
        if len(values) == 2:
            name = default_name
            start = values[0]
            stop = values[1]
        elif len(values) == 3:
            name = str(values[0])
            start = values[1]
            stop = values[2]
        else:
            raise ValueError("windows must be BandFeatureWindow objects, mappings, (start, stop), or (name, start, stop).")

    normalized = BandFeatureWindow(
        name,
        _normalize_window_endpoint(start, name="start", allow_negative_infinity=True),
        _normalize_window_endpoint(stop, name="stop", allow_positive_infinity=True),
    )
    if normalized.start >= normalized.stop:
        raise ValueError("time_window start must be before stop.")
    return normalized


def _normalize_windows(windows) -> tuple[BandFeatureWindow, ...]:
    if windows is None:
        return (BandFeatureWindow("window", -np.inf, np.inf),)
    if isinstance(windows, BandFeatureWindow) or isinstance(windows, Mapping):
        return (_normalize_window(windows),)
    return tuple(_normalize_window(window, default_name=f"window_{index}") for index, window in enumerate(windows))


def _time_mask(time_vector: np.ndarray, time_window) -> np.ndarray:
    window = _normalize_window(time_window)
    mask = (time_vector >= window.start) & (time_vector <= window.stop)
    if not np.any(mask):
        raise ValueError(f"time_window {window.as_tuple} does not overlap the data.")
    return mask


def compute_band_analytic_window(
    signal,
    time_vector,
    *,
    band_hz: Sequence[float] = (8.0, 12.0),
    time_window: Sequence[float] = (-np.inf, np.inf),
    filter_order: int = 5,
    time_axis: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return band-limited analytic signal samples in ``time_window``.

    Parameters
    ----------
    signal:
        Real-valued samples.  The last axis is interpreted as time by default.
    time_vector:
        Sample times in seconds.  Row-vector MATLAB time axes are accepted and
        flattened.
    band_hz:
        Two cutoff frequencies in Hz.
    time_window:
        Inclusive ``(start, stop)`` window in seconds.
    filter_order:
        Butterworth filter order.
    time_axis:
        Axis of ``signal`` that corresponds to ``time_vector``.
    """

    signal, time_vector, sampling_rate, time_axis = _validate_signal_time_axis(signal, time_vector, time_axis=time_axis)
    validate_band_hz(band_hz, sampling_rate)
    time_indices = np.flatnonzero(_time_mask(time_vector, time_window))
    analytic_signal = band_analytic_signal(signal, sampling_rate, band_hz, order=filter_order, axis=time_axis)
    return np.take(analytic_signal, time_indices, axis=time_axis), time_indices


def summarize_analytic_window(
    analytic_window,
    *,
    outputs: Sequence[str] = DEFAULT_BAND_FEATURE_OUTPUTS,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Summarize a complex analytic signal window into scalar features."""

    analytic_window = np.asarray(analytic_window)
    if analytic_window.size == 0:
        raise ValueError("analytic_window must contain at least one sample.")
    if not np.all(np.isfinite(analytic_window.real)) or not np.all(np.isfinite(analytic_window.imag)):
        raise ValueError("analytic_window must contain only finite values.")

    power = np.abs(analytic_window) ** 2
    phase = np.angle(analytic_window)
    output_set = tuple(outputs)
    unknown = set(output_set) - set(DEFAULT_BAND_FEATURE_OUTPUTS)
    if unknown:
        raise ValueError(f"Unknown oscillatory feature output(s): {sorted(unknown)}")

    row: dict[str, float] = {}
    if "mean_power" in output_set:
        row["mean_power"] = float(np.mean(power))
    if "log_power" in output_set:
        row["log_power"] = float(np.mean(np.log(power + eps)))
    if "phase_concentration" in output_set:
        row["phase_concentration"] = float(np.abs(np.mean(np.exp(1j * phase))))
    if "mean_phase" in output_set:
        row["mean_phase"] = float(circular_mean_phase(phase, axis=None))
    return row


def compute_band_trial_features(
    signal,
    time_vector,
    *,
    band_hz: Sequence[float] = (8.0, 12.0),
    time_window: Sequence[float] = (-np.inf, np.inf),
    channel_indices: Sequence[int] | None = None,
    channel_axis: int = 0,
    time_axis: int = -1,
    filter_order: int = 5,
    outputs: Sequence[str] = DEFAULT_BAND_FEATURE_OUTPUTS,
) -> dict[str, float]:
    """Compute scalar band features for one trial/window/channel group."""

    signal = np.asarray(signal, dtype=float)
    if channel_indices is not None:
        if signal.ndim < 2:
            raise ValueError("channel_indices require a signal with a channel axis.")
        channel_axis = _normalize_axis(channel_axis, signal.ndim)
        signal = np.take(signal, np.asarray(channel_indices, dtype=int), axis=channel_axis)
    analytic_window, _ = compute_band_analytic_window(
        signal,
        time_vector,
        band_hz=band_hz,
        time_window=time_window,
        filter_order=filter_order,
        time_axis=time_axis,
    )
    return summarize_analytic_window(analytic_window, outputs=outputs)


def _selected_channel_count(signal: np.ndarray, channel_indices: Sequence[int] | None, channel_axis: int) -> int:
    if channel_indices is not None:
        return int(len(channel_indices))
    if signal.ndim < 2:
        return 1
    return int(signal.shape[channel_axis])


def compute_band_features(
    data,
    time_vector,
    *,
    band_hz: Sequence[float] = (8.0, 12.0),
    windows=None,
    channel_indices: Sequence[int] | None = None,
    labels: Sequence | None = None,
    trial_axis: int = 0,
    channel_axis: int = 1,
    time_axis: int = -1,
    filter_order: int = 5,
    outputs: Sequence[str] = DEFAULT_BAND_FEATURE_OUTPUTS,
    as_dataframe: bool = False,
):
    """Compute band features for a trials-by-channels-by-time array.

    Returns one row per trial and window.  Rows contain trial/window metadata,
    band edges, channel count, optional label, and requested feature columns.
    """

    data = np.asarray(data, dtype=float)
    if data.ndim != 3:
        raise ValueError("data must be a 3D trials-by-channels-by-time array.")
    trial_axis = _normalize_axis(trial_axis, data.ndim)
    channel_axis = _normalize_axis(channel_axis, data.ndim)
    time_axis = _normalize_axis(time_axis, data.ndim)
    if len({trial_axis, channel_axis, time_axis}) != 3:
        raise ValueError("trial_axis, channel_axis, and time_axis must be distinct.")

    time_vector = validate_time_axis(time_vector)
    if data.shape[time_axis] != time_vector.size:
        raise ValueError(
            f"data has {data.shape[time_axis]} samples along its time axis but time_vector has "
            f"{time_vector.size} entries."
        )

    windows = _normalize_windows(windows)
    standard = np.moveaxis(data, (trial_axis, channel_axis, time_axis), (0, 1, 2))
    labels_array = None if labels is None else np.asarray(labels, dtype=object)
    if labels_array is not None and labels_array.shape[0] != standard.shape[0]:
        raise ValueError("labels must contain one value per trial.")

    low_freq, high_freq = validate_band_hz(band_hz, sampling_rate_from_time_axis(time_vector))
    rows: list[dict[str, object]] = []
    for trial_idx, trial_signal in enumerate(standard):
        n_channels = _selected_channel_count(trial_signal, channel_indices, 0)
        for window in windows:
            summary = compute_band_trial_features(
                trial_signal,
                time_vector,
                band_hz=(low_freq, high_freq),
                time_window=window.as_tuple,
                channel_indices=channel_indices,
                channel_axis=0,
                time_axis=-1,
                filter_order=filter_order,
                outputs=outputs,
            )
            row: dict[str, object] = {
                "trial": trial_idx,
                "window": window.name,
                "time_window_start": window.start,
                "time_window_stop": window.stop,
                "low_freq": low_freq,
                "high_freq": high_freq,
                "n_channels": n_channels,
            }
            if labels_array is not None:
                row["label"] = labels_array[trial_idx].item() if hasattr(labels_array[trial_idx], "item") else labels_array[trial_idx]
            row.update(summary)
            rows.append(row)

    if as_dataframe:
        import pandas as pd

        return pd.DataFrame(rows)
    return rows


def compute_alpha_features(data, time_vector, **kwargs):
    """Compute alpha-band features using the default 8--12 Hz band."""

    kwargs.setdefault("band_hz", (8.0, 12.0))
    return compute_band_features(data, time_vector, **kwargs)
