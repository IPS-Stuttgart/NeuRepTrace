"""Runtime guardrail for cue-source temporal binning.

``np.array_split`` returns empty arrays when more temporal bins are requested than
there are response-window samples.  The cue-feature helpers subsequently average
those empty arrays, creating NaNs that the final feature-vector cleanup converts
to zeros.  Rejecting this configuration keeps cue calibration features from
silently carrying degenerate bins.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_bushmeg_cue_temporal_bins_patch_installed"


def _response_temporal_bins(module: ModuleType, times: Any, window: tuple[float, float], temporal_bins: Any) -> int:
    normalized_bins = module._normalize_temporal_bins(temporal_bins)
    mask = module._time_mask(np.asarray(times, dtype=float), window, name="cue response")
    n_response_samples = int(np.count_nonzero(mask))
    if normalized_bins > n_response_samples:
        raise ValueError(
            f"cue_source_weighting.temporal_bins ({normalized_bins}) must not exceed the {n_response_samples} cue response-window sample(s)."
        )
    return normalized_bins


def _patch_module(module: ModuleType) -> None:
    if getattr(module, _PATCH_MARKER, False):
        return

    original_evoked_bin_means = module._evoked_bin_means
    original_evoked_gfp_bins = module._evoked_gfp_bins

    def _evoked_bin_means(data: np.ndarray, times: np.ndarray, window: tuple[float, float], *, temporal_bins: int) -> np.ndarray:
        normalized_bins = _response_temporal_bins(module, times, window, temporal_bins)
        return original_evoked_bin_means(data, times, window, temporal_bins=normalized_bins)

    def _evoked_gfp_bins(data: np.ndarray, times: np.ndarray, window: tuple[float, float], *, temporal_bins: int) -> np.ndarray:
        normalized_bins = _response_temporal_bins(module, times, window, temporal_bins)
        return original_evoked_gfp_bins(data, times, window, temporal_bins=normalized_bins)

    module._evoked_bin_means = _evoked_bin_means
    module._evoked_gfp_bins = _evoked_gfp_bins
    setattr(module, _PATCH_MARKER, True)


def install() -> None:
    """Install empty-response-bin validation for cue-source feature helpers."""

    from neureptrace import bushmeg_cue_source_weights

    _patch_module(bushmeg_cue_source_weights)
