"""Exploratory alpha-band metrics for MEG trials."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.signal

from neureptrace.meg.alpha_signal import uniform_sample_interval
from neureptrace.meg.fieldtrip_struct import count_trials, get_time_vector, get_trial_signal, load_fieldtrip_mat, trial_label
from neureptrace.meg.sensor_geometry import (
    DEFAULT_MIN_REFERENCE_AXIS_PROJECTION,
    DEFAULT_OCCIPITAL_PATTERN,
    DEFAULT_PROJECTION_REFERENCE_PATTERN,
    DEFAULT_SENSOR_POSITION_UNIT,
    delaunay_edges,
    project_channel_positions,
    select_channels,
)

DEFAULT_TIME_WINDOW = (-0.4, -0.05)
DEFAULT_FREQUENCY_RANGE = (8.0, 12.0)


@dataclass(frozen=True)
class AlphaMetricConfig:
    """Parameters controlling alpha metric extraction."""

    location_pattern: str = DEFAULT_OCCIPITAL_PATTERN
    time_window: tuple[float, float] = DEFAULT_TIME_WINDOW
    frequency_range: tuple[float, float] = DEFAULT_FREQUENCY_RANGE
    filter_order: int = 5
    sensor_position_unit: str = DEFAULT_SENSOR_POSITION_UNIT
    projection_reference_pattern: str | None = DEFAULT_PROJECTION_REFERENCE_PATTERN
    min_reference_axis_projection: float = DEFAULT_MIN_REFERENCE_AXIS_PROJECTION


def _time_mask(time_vector, time_window: tuple[float, float]) -> np.ndarray:
    start, stop = time_window
    if start >= stop:
        raise ValueError("time_window start must be before stop.")
    mask = (time_vector >= start) & (time_vector <= stop)
    if not np.any(mask):
        raise ValueError(f"time_window {time_window} does not overlap the data.")
    return mask


def _validate_alpha_signal_time_axis(signal, time_vector) -> tuple[np.ndarray, np.ndarray, float]:
    signal = np.asarray(signal, dtype=float)
    time_vector = np.asarray(time_vector, dtype=float).ravel()
    if signal.ndim == 0:
        raise ValueError("signal must have at least one time dimension.")
    if signal.shape[-1] != time_vector.size:
        raise ValueError(f"signal has {signal.shape[-1]} samples along its last axis but time_vector has {time_vector.size} entries.")
    sample_interval = uniform_sample_interval(time_vector)
    return signal, time_vector, sample_interval


def compute_alpha_analytic_window(signal, time_vector, config: AlphaMetricConfig):
    """Return alpha-band analytic signal samples in ``config.time_window``."""

    signal, time_vector, sample_interval = _validate_alpha_signal_time_axis(signal, time_vector)
    sampling_rate = float(1 / sample_interval)
    time_indices = np.flatnonzero(_time_mask(time_vector, config.time_window))
    low_freq, high_freq = config.frequency_range

    sos = scipy.signal.butter(config.filter_order, [low_freq, high_freq], btype="bandpass", fs=sampling_rate, output="sos")
    alpha_signal = scipy.signal.sosfiltfilt(sos, signal, axis=-1)
    analytic_signal = scipy.signal.hilbert(alpha_signal, axis=-1)
    return np.take(analytic_signal, time_indices, axis=-1), time_indices


def _alpha_window_and_phase(signal, time_vector, config: AlphaMetricConfig) -> tuple[np.ndarray, np.ndarray]:
    alpha_window, _ = compute_alpha_analytic_window(signal, time_vector, config)
    return alpha_window, np.angle(alpha_window)


def _phase_geometry(data, channel_indices, config: AlphaMetricConfig):
    _, coords2d = project_channel_positions(
        data,
        channel_indices,
        sensor_position_unit=config.sensor_position_unit,
        projection_reference_pattern=config.projection_reference_pattern,
        min_reference_axis_projection=config.min_reference_axis_projection,
    )
    return delaunay_edges(coords2d)


def _phase_gradient_metrics(phase, edge_indices, edge_vectors, edge_pinv, center_frequency: float) -> dict[str, float]:
    phase_delta = np.angle(np.exp(1j * (phase[edge_indices[:, 1], :] - phase[edge_indices[:, 0], :])))
    gradients = edge_pinv @ phase_delta
    predicted_delta = edge_vectors @ gradients
    residual = np.angle(np.exp(1j * (phase_delta - predicted_delta)))
    fit = np.abs(np.mean(np.exp(1j * residual), axis=0))
    gradient_norm = np.linalg.norm(gradients, axis=0)
    weights = fit + 1e-12
    mean_gradient = np.average(gradients, axis=1, weights=weights)
    valid = (fit > 0.5) & (gradient_norm > 1e-4)

    speed_m_per_s = np.nan
    if np.any(valid):
        speed_m_per_s = np.nanmedian((2 * np.pi * center_frequency / gradient_norm[valid]) / 1000.0)

    return {
        "phase_plane_fit": float(np.mean(fit)),
        "spatial_freq_rad_per_mm": float(np.average(gradient_norm, weights=weights)),
        "speed_m_per_s": float(speed_m_per_s),
        "gradient_x": float(mean_gradient[0]),
        "gradient_y": float(mean_gradient[1]),
        "direction_rad": float(np.arctan2(mean_gradient[1], mean_gradient[0])),
    }


def _resolve_channel_indices(data, channel_indices, config: AlphaMetricConfig) -> np.ndarray:
    if channel_indices is None:
        channel_indices = select_channels(data, config.location_pattern)
    channel_indices = np.asarray(channel_indices, dtype=int)
    if channel_indices.size == 0:
        raise ValueError(f"No channels matched pattern: {config.location_pattern}")
    return channel_indices


def compute_alpha_trial_metrics(
    data,
    trial_idx: int,
    *,
    participant_id: Any = None,
    dataset: str = "main",
    channel_indices=None,
    config: AlphaMetricConfig | None = None,
) -> dict[str, Any]:
    """Compute exploratory prestimulus alpha metrics for one trial."""

    config = config or AlphaMetricConfig()
    channel_indices = _resolve_channel_indices(data, channel_indices, config)
    time_vector = get_time_vector(data, trial_idx)
    signal = np.take(get_trial_signal(data, trial_idx), channel_indices, axis=0)
    alpha_window, phase = _alpha_window_and_phase(signal, time_vector, config)
    edge_indices, edge_vectors, edge_pinv = _phase_geometry(data, channel_indices, config)

    row = {
        "participant": participant_id if participant_id is not None else "",
        "dataset": dataset,
        "trial": trial_idx,
        "trial_label": trial_label(data, trial_idx),
        "time_window_start": config.time_window[0],
        "time_window_stop": config.time_window[1],
        "low_freq": config.frequency_range[0],
        "high_freq": config.frequency_range[1],
        "n_channels": int(len(channel_indices)),
        "alpha_power": float(np.mean(np.abs(alpha_window) ** 2)),
        "log_alpha_power": float(np.mean(np.log(np.abs(alpha_window) ** 2 + 1e-12))),
        "phase_concentration": float(np.abs(np.mean(np.exp(1j * phase)))),
    }
    row.update(_phase_gradient_metrics(phase, edge_indices, edge_vectors, edge_pinv, center_frequency=sum(config.frequency_range) / 2))
    return row


def compute_alpha_metrics(
    data,
    *,
    participant_id: Any = None,
    dataset: str = "main",
    channel_indices=None,
    config: AlphaMetricConfig | None = None,
) -> list[dict[str, Any]]:
    """Compute alpha metrics for every trial in ``data``."""

    config = config or AlphaMetricConfig()
    channel_indices = _resolve_channel_indices(data, channel_indices, config)
    return [
        compute_alpha_trial_metrics(data, trial_idx, participant_id=participant_id, dataset=dataset, channel_indices=channel_indices, config=config)
        for trial_idx in range(count_trials(data))
    ]


def write_alpha_metrics_csv(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    """Write alpha metric rows to ``output_path``."""

    if not rows:
        raise ValueError("At least one row is required.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_participant_data(data_folder, participant_id, *, cue: bool = False, file_pattern: str | None = None, root_path=None):
    """Load a participant's main or cue MATLAB data file.

    This compatibility helper keeps the reusable implementation usable by the
    existing PyMEGDec command style while allowing dataset-specific callers to
    pass a custom ``file_pattern`` or bypass it entirely with ``load_fieldtrip_mat``.
    """

    suffix = "CueData" if cue else "Data"
    if file_pattern is None:
        file_pattern = "Part{participant}{suffix}.mat"
    data_path = Path(data_folder) / file_pattern.format(participant=participant_id, suffix=suffix, cue="Cue" if cue else "")
    return load_fieldtrip_mat(data_path, root_path=root_path)


def export_participant_alpha_metrics(
    data_folder,
    participant_id,
    output_path,
    *,
    cue: bool = False,
    config: AlphaMetricConfig | None = None,
    file_pattern: str | None = None,
    root_path=None,
):
    """Load participant data, compute alpha metrics, and write a CSV."""

    config = config or AlphaMetricConfig()
    data = load_participant_data(data_folder, participant_id, cue=cue, file_pattern=file_pattern, root_path=root_path)
    rows = compute_alpha_metrics(data, participant_id=participant_id, dataset="cue" if cue else "main", config=config)
    write_alpha_metrics_csv(rows, output_path)
    return rows
