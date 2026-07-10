"""Runtime guardrails for MMD gamma validation and stable distances.

The core MMD helper accepts numeric gamma values and the named heuristics
``median``/``auto``/``scale``.  Without the gamma guard, an arbitrary string
falls through to ``float(...)`` and raises Python's low-level conversion error.

The core squared-distance identity ``||x||^2 + ||y||^2 - 2 x^T y`` can also
produce ``NaN`` through ``inf - inf`` for finite, large-magnitude features.
Using SciPy's direct squared-Euclidean implementation avoids that cancellation:
identical large vectors remain at distance zero, while genuinely unrepresentable
squared distances saturate to infinity and therefore yield an RBF kernel of zero.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_mmd_gamma_validation_patch_installed"
_DISTANCE_PATCH_MARKER = "_neureptrace_mmd_stable_distance_patch_installed"
_GAMMA_ERROR = "gamma must be positive and finite, or one of: median, auto, scale."
_KNOWN_STRING_GAMMAS = {"median", "auto", "median_distance", "median_heuristic", "scale"}


def _is_numeric_string(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def install() -> None:
    """Install strict gamma validation and cancellation-safe MMD distances."""

    import neureptrace.decoding.mmd_source_weighting as mmd_source_weighting

    if not getattr(mmd_source_weighting.resolve_mmd_gamma, _PATCH_MARKER, False):
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

    if not getattr(mmd_source_weighting._squared_euclidean, _DISTANCE_PATCH_MARKER, False):
        original_squared_euclidean = mmd_source_weighting._squared_euclidean

        @wraps(original_squared_euclidean)
        def _squared_euclidean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
            from scipy.spatial.distance import cdist

            left_matrix = np.asarray(left, dtype=np.float64)
            right_matrix = np.asarray(right, dtype=np.float64)
            with np.errstate(over="ignore", invalid="ignore"):
                squared = cdist(left_matrix, right_matrix, metric="sqeuclidean")
            return np.maximum(squared, 0.0)

        setattr(_squared_euclidean, _DISTANCE_PATCH_MARKER, True)
        mmd_source_weighting._squared_euclidean = _squared_euclidean


__all__ = ["install"]
