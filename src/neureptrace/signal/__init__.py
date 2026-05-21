"""Reusable signal-processing primitives for NeuRepTrace."""

from __future__ import annotations

from neureptrace.signal.band import (
    average_phases,
    band_analytic_signal,
    bandpass_filter,
    bandpass_filter_signal,
    bandpass_sos,
    circular_mean_phase,
    extract_alpha_signal_and_phase,
    extract_band_signal_and_phase,
    extract_phase,
    sampling_rate_from_time_axis,
    sampling_rate_from_time_vector,
    uniform_sample_interval,
    validate_band_hz,
    validate_sampling_rate,
    validate_signal_values,
    validate_time_axis,
)

__all__ = [
    "average_phases",
    "band_analytic_signal",
    "bandpass_filter",
    "bandpass_filter_signal",
    "bandpass_sos",
    "circular_mean_phase",
    "extract_alpha_signal_and_phase",
    "extract_band_signal_and_phase",
    "extract_phase",
    "sampling_rate_from_time_axis",
    "sampling_rate_from_time_vector",
    "uniform_sample_interval",
    "validate_band_hz",
    "validate_sampling_rate",
    "validate_signal_values",
    "validate_time_axis",
]
