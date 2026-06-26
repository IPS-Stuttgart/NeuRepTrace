"""MMD-based source-group weighting for unlabeled OpenNeuro target folds.

This module is intentionally protocol-explicit.  It scores each candidate
source group against the unlabeled held-out target feature distribution with a
biased RBF maximum mean discrepancy (MMD) estimate, then converts the negative
MMD values to mean-one source-group multipliers.  It uses source features and
unlabeled target features only; held-out target labels are neither accepted nor
needed.

The intended use is Protocol 2/2.5 style OpenNeuro transfer experiments where
strict source-only training underperforms because the held-out dataset/subject
has a different feature distribution, but using target labels would be leakage.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

MMD_SOURCE_WEIGHTING_PROTOCOL = "unlabeled_target_mmd_source_weighting"
MMD_SOURCE_WEIGHTING_PROTOCOL_CATEGORY = "2.5_unlabeled_target_distribution_adaptive"
DEFAULT_MMD_TEMPERATURE = 0.25
DEFAULT_MMD_GAMMA: float | str = "median"
_MIN_SCALE = 1.0e-12
_GAMMA_ERROR = "gamma must be positive and finite, or one of: median, auto, scale."


@dataclass(frozen=True, slots=True)
class MMDSourceWeightingResult:
    """MMD source-group weights, scores, and leakage-provenance metadata."""

    weights: dict[Hashable, float]
    mmd_squared: dict[Hashable, float]
    scores: dict[Hashable, float]
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments

def mmd_source_group_weights(
    source_features: Mapping[Hashable, Sequence[Sequence[float]] | Sequence[float] | np.ndarray],
    target_features: Sequence[Sequence[float]] | Sequence[float] | np.ndarray,
    *,
    groups: Sequence[Hashable] | None = None,
    gamma: float | str = DEFAULT_MMD_GAMMA,
    temperature: float = DEFAULT_MMD_TEMPERATURE,
    top_k: int | str | None = None,
    blend: float = 1.0,
) -> MMDSourceWeightingResult:
    """Return mean-one source-group weights from unlabeled target MMD.

    Parameters
    ----------
    source_features:
        Mapping from source group identifier to feature rows for that source.
    target_features:
        Unlabeled target feature rows.  Target labels are intentionally not part
        of this API.
    groups:
        Optional explicit group order/subset.  Duplicate group names are removed
        in first-observed order.
    gamma:
        RBF kernel width.  ``"median"``/``"auto"`` uses all supplied source and
        target rows to compute the median squared-distance heuristic.
        ``"scale"`` uses ``1 / n_features``.
    temperature:
        Softmax temperature applied to negative MMD values.  Lower values make
        source selection sharper.
    top_k:
        Optional number of lowest-MMD source groups to keep.  Other groups get
        zero pre-normalization mass.  String aliases such as ``"none"`` and
        integer-like strings are accepted for config-driven runs.
    blend:
        Convex blend between uniform weights and MMD weights.  ``0`` is uniform;
        ``1`` is the full MMD weighting.
    """

    normalized_top_k = _optional_positive_int(top_k, name="top_k")
    group_list = _group_list(groups, source_features)
    if not group_list:
        return MMDSourceWeightingResult(weights={}, mmd_squared={}, scores={}, metadata=_metadata(0, 0, 0, gamma, temperature, normalized_top_k, blend))
    target = _feature_matrix(target_features, name="target_features")
    source_matrices = {group: _feature_matrix(source_features[group], name=f"source_features[{group!r}]") for group in group_list}
    feature_dim = target.shape[1]
    for group, matrix in source_matrices.items():
        if matrix.shape[1] != feature_dim:
            raise ValueError(f"source_features[{group!r}] has feature width {matrix.shape[1]}, expected {feature_dim}.")

    gamma_value = resolve_mmd_gamma(gamma, source_matrices.values(), target)
    target_kernel_mean = _kernel_mean(target, target, gamma_value)
    mmd_squared: dict[Hashable, float] = {}
    for group in group_list:
        source = source_matrices[group]
        mmd = _kernel_mean(source, source, gamma_value) + target_kernel_mean - 2.0 * _kernel_mean(source, target, gamma_value)
        mmd_squared[group] = max(0.0, float(mmd))

    scores = {group: -value for group, value in mmd_squared.items()}
    weights = _weights_from_scores(scores, group_list, temperature=temperature, top_k=normalized_top_k, blend=blend)
    metadata = _metadata(
        len(group_list),
        int(target.shape[0]),
        int(feature_dim),
        gamma_value,
        temperature,
        normalized_top_k,
        blend,
    )
    metadata.update(
        {
            "mmd_best_group": min(mmd_squared, key=mmd_squared.get),
            "mmd_best_value": float(min(mmd_squared.values())),
            "mmd_worst_value": float(max(mmd_squared.values())),
            "mmd_weight_mean": float(np.mean(list(weights.values()))),
            "mmd_weight_max": float(np.max(list(weights.values()))),
            "mmd_weight_min": float(np.min(list(weights.values()))),
        }
    )
    return MMDSourceWeightingResult(weights=weights, mmd_squared=mmd_squared, scores=scores, metadata=metadata)


def mmd_source_group_scores(
    source_features: Mapping[Hashable, Sequence[Sequence[float]] | Sequence[float] | np.ndarray],
    target_features: Sequence[Sequence[float]] | Sequence[float] | np.ndarray,
    *,
    groups: Sequence[Hashable] | None = None,
    gamma: float | str = DEFAULT_MMD_GAMMA,
) -> dict[Hashable, float]:
    """Return negative-MMD utility scores for source groups."""

    return mmd_source_group_weights(
        source_features,
        target_features,
        groups=groups,
        gamma=gamma,
        temperature=1.0,
        blend=1.0,
    ).scores


def resolve_mmd_gamma(
    value: float | str,
    source_feature_matrices: Sequence[np.ndarray],
    target_features: Sequence[Sequence[float]] | np.ndarray,
) -> float:
    """Resolve an RBF gamma value for MMD scoring."""

    target = _feature_matrix(target_features, name="target_features")
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(_GAMMA_ERROR)
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_")
        if normalized in {"median", "auto", "median_distance", "median_heuristic"}:
            matrices = [np.asarray(matrix, dtype=float) for matrix in source_feature_matrices]
            matrices.append(target)
            stacked = np.vstack(matrices)
            squared = _upper_pairwise_squared_distances(stacked)
            positive = squared[squared > _MIN_SCALE]
            sigma2 = float(np.median(positive)) if positive.size else 1.0
            return 1.0 / (2.0 * max(sigma2, _MIN_SCALE))
        if normalized == "scale":
            return 1.0 / max(1, target.shape[1])
        try:
            numeric = float(normalized)
        except ValueError as exc:
            raise ValueError(_GAMMA_ERROR) from exc
    else:
        numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(_GAMMA_ERROR)
    return numeric


def _weights_from_scores(
    scores: Mapping[Hashable, float],
    groups: Sequence[Hashable],
    *,
    temperature: float,
    top_k: int | str | None,
    blend: float,
) -> dict[Hashable, float]:
    temperature = _positive_float(temperature, name="temperature")
    blend = _unit_interval_float(blend, name="blend")
    normalized_top_k = _optional_positive_int(top_k, name="top_k")
    utilities = np.asarray([float(scores[group]) for group in groups], dtype=float)
    utilities = _replace_nonfinite_with_minimum(utilities)
    keep = np.ones(len(groups), dtype=bool)
    if normalized_top_k is not None and normalized_top_k < len(groups):
        keep[:] = False
        keep[np.argsort(utilities)[-normalized_top_k:]] = True
    raw = np.zeros(len(groups), dtype=float)
    if np.any(keep):
        kept = utilities[keep]
        raw[keep] = np.exp(np.clip((kept - float(np.max(kept))) / temperature, -60.0, 0.0))
    weights = _mean_one(raw)
    if blend < 1.0:
        weights = _mean_one((1.0 - blend) * np.ones_like(weights) + blend * weights)
    return {group: float(weight) for group, weight in zip(groups, weights, strict=True)}


def _group_list(groups: Sequence[Hashable] | None, source_features: Mapping[Hashable, Any]) -> list[Hashable]:
    if groups is not None:
        group_list = list(dict.fromkeys(groups))
        missing = [group for group in group_list if group not in source_features]
        if missing:
            raise ValueError(f"Missing source features for groups: {missing}.")
        return group_list
    return list(source_features.keys())


def _feature_matrix(features: Sequence[Sequence[float]] | Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a one- or two-dimensional array.")
    if matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one row and one feature.")
    return np.nan_to_num(matrix.astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0)


def _kernel_mean(left: np.ndarray, right: np.ndarray, gamma: float) -> float:
    return float(np.mean(np.exp(-float(gamma) * _squared_euclidean(left, right))))


def _squared_euclidean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.sum(left * left, axis=1, keepdims=True)
    right_norm = np.sum(right * right, axis=1, keepdims=True).T
    return np.maximum(left_norm + right_norm - 2.0 * (left @ right.T), 0.0)


def _upper_pairwise_squared_distances(values: np.ndarray) -> np.ndarray:
    if values.shape[0] <= 1:
        return np.asarray([1.0], dtype=float)
    squared = _squared_euclidean(values, values)
    return squared[np.triu_indices(values.shape[0], k=1)]


def _replace_nonfinite_with_minimum(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if np.all(finite):
        return values
    if not np.any(finite):
        return np.zeros_like(values)
    replacement = float(np.min(values[finite]))
    return np.where(finite, values, replacement)


def _mean_one(weights: np.ndarray, *, epsilon: float = 1e-12) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    if np.any(weights < 0.0):
        raise ValueError("Weights must be non-negative.")
    mean = float(weights.mean()) if weights.size else 1.0
    if mean <= epsilon:
        return np.ones_like(weights)
    return weights / mean


def _positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be positive and finite.")
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return number


def _unit_interval_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be in [0, 1].")
    number = float(value)
    if not np.isfinite(number) or number < 0.0 or number > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return number


def _optional_positive_int(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "all", "full"}:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer or null.")
    number = float(value)
    if not np.isfinite(number) or number % 1.0 != 0.0 or number < 1:
        raise ValueError(f"{name} must be a positive integer or null.")
    return int(number)


def _metadata(n_groups: int, n_target: int, feature_dim: int, gamma: float | str, temperature: float, top_k: int | None, blend: float) -> dict[str, Any]:
    return {
        "mmd_source_weighting_protocol": MMD_SOURCE_WEIGHTING_PROTOCOL,
        "mmd_source_weighting_protocol_category": MMD_SOURCE_WEIGHTING_PROTOCOL_CATEGORY,
        "mmd_source_weighting_uses_unlabeled_target_data": True,
        "mmd_source_weighting_uses_target_labels": False,
        "mmd_source_weighting_valid_for_protocol_2_5": True,
        "mmd_source_weighting_valid_for_strict_source_only": False,
        "mmd_n_source_groups": int(n_groups),
        "mmd_n_target_rows": int(n_target),
        "mmd_feature_dim": int(feature_dim),
        "mmd_gamma": float(gamma) if isinstance(gamma, (float, int, np.floating, np.integer)) else gamma,
        "mmd_temperature": float(temperature),
        "mmd_top_k": "" if top_k is None else int(top_k),
        "mmd_blend": float(blend),
    }
