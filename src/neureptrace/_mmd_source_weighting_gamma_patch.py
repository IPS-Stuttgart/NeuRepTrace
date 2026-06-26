"""Runtime guardrail for MMD gamma string validation.

The core MMD helper accepts numeric gamma values and the named heuristics
``median``/``auto``/``scale``.  Without this guard, an arbitrary string falls
through to ``float(...)`` and raises Python's low-level conversion error instead
of the public validation message used for the other invalid gamma values.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_mmd_gamma_validation_patch_installed"
_GAMMA_ERROR = "gamma must be positive and finite, or one of: median, auto, scale."
_KNOWN_STRING_GAMMAS = {"median", "auto", "median_distance", "median_heuristic", "scale"}


def _is_numeric_string(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def install() -> None:
    """Install strict user-facing validation for string MMD gamma values."""

    import neureptrace.decoding.mmd_source_weighting as mmd_source_weighting

    if getattr(mmd_source_weighting.resolve_mmd_gamma, _PATCH_MARKER, False):
        return

    original_resolve_mmd_gamma = mmd_source_weighting.resolve_mmd_gamma

    @wraps(original_resolve_mmd_gamma)
    def resolve_mmd_gamma(value: Any, source_feature_matrices, target_features) -> float:
        if isinstance(value, str):
            normalized = value.strip().lower().replace("-", "_")
            if normalized not in _KNOWN_STRING_GAMMAS and not _is_numeric_string(normalized):
                raise ValueError(_GAMMA_ERROR)
        elif isinstance(value, (bool, np.bool_)):
            raise ValueError(_GAMMA_ERROR)
        return original_resolve_mmd_gamma(value, source_feature_matrices, target_features)

    setattr(resolve_mmd_gamma, _PATCH_MARKER, True)
    mmd_source_weighting.resolve_mmd_gamma = resolve_mmd_gamma


__all__ = ["install"]
