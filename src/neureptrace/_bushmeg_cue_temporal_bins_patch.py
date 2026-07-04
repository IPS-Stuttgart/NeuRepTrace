"""Runtime guardrail for cue-source temporal binning.

``np.array_split`` returns empty arrays when more temporal bins are requested than
there are response-window samples.  The cue-feature helpers subsequently average
those empty arrays, creating NaNs that the final feature-vector cleanup converts
to zeros.  Rejecting this configuration keeps cue calibration features from
silently carrying degenerate bins.
"""

from __future__ import annotations

import math
from functools import wraps
from types import ModuleType
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_bushmeg_cue_temporal_bins_patch_installed"
_LOSO_PATCH_MARKER = "_neureptrace_bushmeg_loso_max_folds_patch_installed"


def _response_temporal_bins(module: ModuleType, times: Any, window: tuple[float, float], temporal_bins: Any) -> int:
    normalized_bins = module._normalize_temporal_bins(temporal_bins)
    mask = module._time_mask(np.asarray(times, dtype=float), window, name="cue response")
    n_response_samples = int(np.count_nonzero(mask))
    if normalized_bins > n_response_samples:
        raise ValueError(
            f"cue_source_weighting.temporal_bins ({normalized_bins}) must not exceed the {n_response_samples} cue response-window sample(s)."
        )
    return normalized_bins


def _non_negative_max_folds(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("max_folds must be a non-negative integer or None.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_folds must be a non-negative integer or None.") from exc
    if not math.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 0.0:
        raise ValueError("max_folds must be a non-negative integer or None.")
    return int(parsed)


def _patch_loso_module(module: ModuleType) -> None:
    if getattr(module, _LOSO_PATCH_MARKER, False):
        return

    original_run = module.run_bushmeg_loso_decode

    @wraps(original_run)
    def run_bushmeg_loso_decode(*args, **kwargs):
        if "max_folds" in kwargs:
            kwargs = dict(kwargs)
            kwargs["max_folds"] = _non_negative_max_folds(kwargs["max_folds"])
        return original_run(*args, **kwargs)

    module._non_negative_max_folds = _non_negative_max_folds
    module.run_bushmeg_loso_decode = run_bushmeg_loso_decode
    setattr(module, _LOSO_PATCH_MARKER, True)


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

    from neureptrace import bushmeg_cue_source_weights, bushmeg_loso_decode

    _patch_loso_module(bushmeg_loso_decode)
    _patch_module(bushmeg_cue_source_weights)
