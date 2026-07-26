"""Feature extraction helpers for NeuRepTrace."""

from __future__ import annotations

from . import (
    _oscillatory_channel_indices_patch,
    _oscillatory_composite_labels_patch,
    _oscillatory_single_window_patch,
)
from . import oscillatory as _oscillatory

_oscillatory_composite_labels_patch.install()
_oscillatory_single_window_patch.install()
_oscillatory_channel_indices_patch.install()

BandFeatureWindow = _oscillatory.BandFeatureWindow
compute_alpha_features = _oscillatory.compute_alpha_features
compute_band_analytic_window = _oscillatory.compute_band_analytic_window
compute_band_features = _oscillatory.compute_band_features
compute_band_trial_features = _oscillatory.compute_band_trial_features
summarize_analytic_window = _oscillatory.summarize_analytic_window

__all__ = [
    "BandFeatureWindow",
    "compute_alpha_features",
    "compute_band_analytic_window",
    "compute_band_features",
    "compute_band_trial_features",
    "summarize_analytic_window",
]
