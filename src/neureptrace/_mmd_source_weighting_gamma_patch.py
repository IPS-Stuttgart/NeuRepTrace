"""Runtime guardrails for MMD gamma validation and stable distances.

The MMD helpers accept numeric gamma values and the named heuristics
``median``/``auto``/``scale``. Without the gamma guard, an arbitrary string
falls through to ``float(...)`` and raises Python's low-level conversion error.

Their core squared-distance identity ``||x||^2 + ||y||^2 - 2 x^T y`` can also
produce ``NaN`` through ``inf - inf`` for finite, large-magnitude features.
Using SciPy's direct squared-Euclidean implementation avoids that cancellation.
For source-domain selection, all rows are first divided by one common finite
scale. This leaves the median-heuristic RBF kernel unchanged while ensuring
that even mathematically unrepresentable raw squared distances remain finite.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_mmd_gamma_validation_patch_installed"
_DISTANCE_PATCH_MARKER = "_neureptrace_mmd_stable_distance_patch_installed"
_SOURCE_SELECTION_DISTANCE_PATCH_MARKER = "_neureptrace_source_selection_mmd_stable_distance_patch_installed"
_GAMMA_ERROR = "gamma must be positive and finite, or one of: median, auto, scale."
_KNOWN_STRING_GAMMAS = {"median", "auto", "median_distance", "median_heuristic", "scale"}
_MIN_SCALE = 1.0e-12


def _is_numeric_string(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _stable_squared_euclidean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    from scipy.spatial.distance import cdist

    left_matrix = np.asarray(left, dtype=np.float64)
    right_matrix = np.asarray(right, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        squared = cdist(left_matrix, right_matrix, metric="sqeuclidean")
    return np.maximum(squared, 0.0)


def _scaled_source_selection_mmd(source: np.ndarray, target: np.ndarray) -> float:
    source_matrix = np.asarray(source, dtype=np.float64)
    target_matrix = np.asarray(target, dtype=np.float64)
    scale = max(float(np.max(np.abs(source_matrix))), float(np.max(np.abs(target_matrix))))
    if scale > 0.0:
        source_matrix = source_matrix / scale
        target_matrix = target_matrix / scale

    combined = np.vstack([source_matrix, target_matrix])
    squared = _stable_squared_euclidean(combined, combined)
    upper = squared[np.triu_indices(combined.shape[0], k=1)]
    positive = upper[upper > 0.0]
    sigma2 = float(np.median(positive)) if positive.size else 1.0
    gamma = 1.0 / (2.0 * max(sigma2, _MIN_SCALE))

    source_kernel = np.exp(-gamma * _stable_squared_euclidean(source_matrix, source_matrix))
    target_kernel = np.exp(-gamma * _stable_squared_euclidean(target_matrix, target_matrix))
    cross_kernel = np.exp(-gamma * _stable_squared_euclidean(source_matrix, target_matrix))
    mmd2 = float(np.mean(source_kernel) + np.mean(target_kernel) - 2.0 * np.mean(cross_kernel))
    return float(np.sqrt(max(0.0, mmd2)))


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
            return _stable_squared_euclidean(left, right)

        setattr(_squared_euclidean, _DISTANCE_PATCH_MARKER, True)
        mmd_source_weighting._squared_euclidean = _squared_euclidean

    import neureptrace.decoding.source_selection as source_selection

    if not getattr(source_selection._domain_distance, _SOURCE_SELECTION_DISTANCE_PATCH_MARKER, False):
        original_domain_distance = source_selection._domain_distance

        @wraps(original_domain_distance)
        def _domain_distance(source: np.ndarray, target: np.ndarray, *, metric: str) -> float:
            if metric == "mmd":
                return _scaled_source_selection_mmd(source, target)
            return original_domain_distance(source, target, metric=metric)

        setattr(_domain_distance, _SOURCE_SELECTION_DISTANCE_PATCH_MARKER, True)
        source_selection._domain_distance = _domain_distance


__all__ = ["install"]
