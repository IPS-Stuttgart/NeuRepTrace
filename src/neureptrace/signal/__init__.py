"""Reusable signal-processing primitives for NeuRepTrace."""

from __future__ import annotations

import importlib

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

importlib.import_module("neureptrace._signal_sampling_rate_finite_patch").install()
_patched_band = importlib.import_module("neureptrace.signal.band")
sampling_rate_from_time_axis = _patched_band.sampling_rate_from_time_axis
sampling_rate_from_time_vector = _patched_band.sampling_rate_from_time_vector

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
