"""Unlabeled target feature normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from collections.abc import Sequence

import numpy as np

ADAPTIVE_FEATURE_NORM_PROTOCOL = "unlabeled_target_adaptive_feature_norm"
ADAPTIVE_FEATURE_NORM_CATEGORY = "2_unlabeled_target_adaptive"
ADAPTIVE_FEATURE_NORM_METHODS = ("none", "target_center", "target_zscore", "domain_zscore", "moment_match")
_MIN_SCALE = 1e-12


@dataclass(frozen=True, slots=True)
class AdaptiveFeatureNormResult:
    train_features: np.ndarray
    test_features: np.ndarray
    source_mean: np.ndarray
    source_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def adaptive_feature_normalize(
    train_features: Sequence[Sequence[float]] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    *,
    method: str = "target_zscore",
    scale_floor: float | str = _MIN_SCALE,
) -> AdaptiveFeatureNormResult:
    """Normalize source/train and held-out target/test feature rows.

    Non-``none`` methods use unlabeled target feature statistics and no target labels.
    """

    method_name = normalize_adaptive_feature_norm_method(method)
    floor = _positive_float(scale_floor, name="scale_floor")
    train = _feature_matrix(train_features, name="train_features")
    test = _feature_matrix(test_features, name="test_features")
    if train.shape[1] != test.shape[1]:
        raise ValueError("train_features and test_features must have the same feature width.")
    source_mean, source_scale = _mean_scale(train, floor)
    target_mean, target_scale = _mean_scale(test, floor)
    if method_name == "none":
        train_out = train.copy()
        test_out = test.copy()
    elif method_name == "target_center":
        train_out = train - target_mean
        test_out = test - target_mean
    elif method_name == "target_zscore":
        train_out = (train - target_mean) / target_scale
        test_out = (test - target_mean) / target_scale
    elif method_name == "domain_zscore":
        train_out = (train - source_mean) / source_scale
        test_out = (test - target_mean) / target_scale
    elif method_name == "moment_match":
        train_out = ((train - source_mean) / source_scale) * target_scale + target_mean
        test_out = test.copy()
    else:  # pragma: no cover
        raise AssertionError(method_name)
    uses_target = method_name != "none"
    metadata = {
        "adaptive_feature_norm": uses_target,
        "adaptive_feature_norm_protocol": ADAPTIVE_FEATURE_NORM_PROTOCOL if uses_target else "strict_source_only",
        "adaptive_feature_norm_protocol_category": ADAPTIVE_FEATURE_NORM_CATEGORY if uses_target else "1_strict_source_only",
        "adaptive_feature_norm_method": method_name,
        "adaptive_feature_norm_uses_source_features": True,
        "adaptive_feature_norm_uses_target_features": uses_target,
        "adaptive_feature_norm_uses_target_labels": False,
        "adaptive_feature_norm_valid_for_strict_source_only": not uses_target,
        "adaptive_feature_norm_valid_for_unlabeled_target_adaptation": True,
        "adaptive_feature_norm_n_train_rows": int(train.shape[0]),
        "adaptive_feature_norm_n_test_rows": int(test.shape[0]),
        "adaptive_feature_norm_feature_dim": int(train.shape[1]),
        "adaptive_feature_norm_scale_floor": float(floor),
        "adaptive_feature_norm_test_mean_norm_after": float(np.linalg.norm(np.mean(test_out, axis=0))),
    }
    return AdaptiveFeatureNormResult(
        train_features=train_out.astype(np.float32, copy=False),
        test_features=test_out.astype(np.float32, copy=False),
        source_mean=source_mean.astype(np.float32, copy=False),
        source_scale=source_scale.astype(np.float32, copy=False),
        target_mean=target_mean.astype(np.float32, copy=False),
        target_scale=target_scale.astype(np.float32, copy=False),
        metadata=metadata,
    )


def normalize_adaptive_feature_norm_method(method: str | None) -> str:
    normalized = "target_zscore" if method is None else str(method).strip().lower().replace("-", "_")
    normalized = {
        "off": "none",
        "identity": "none",
        "target_mean": "target_center",
        "target_standardize": "target_zscore",
        "adabn": "domain_zscore",
        "adaptive_batch_norm": "domain_zscore",
        "source_to_target": "moment_match",
        "source_to_target_moment_match": "moment_match",
    }.get(normalized, normalized)
    if normalized not in ADAPTIVE_FEATURE_NORM_METHODS:
        raise ValueError(f"Unknown adaptive feature normalization method {method!r}.")
    return normalized


def apply_adaptive_feature_norm(features: Sequence[Sequence[float]] | np.ndarray, *, mean: Sequence[float] | np.ndarray, scale: Sequence[float] | np.ndarray | None = None) -> np.ndarray:
    matrix = _feature_matrix(features, name="features")
    loc = np.asarray(mean, dtype=float).reshape(-1)
    if loc.shape[0] != matrix.shape[1]:
        raise ValueError("mean length must match feature width.")
    if scale is None:
        return (matrix - loc).astype(np.float32, copy=False)
    scl = np.asarray(scale, dtype=float).reshape(-1)
    if scl.shape[0] != matrix.shape[1] or np.any(scl <= 0.0) or not np.all(np.isfinite(scl)):
        raise ValueError("scale must be finite, positive, and match feature width.")
    return ((matrix - loc) / scl).astype(np.float32, copy=False)


def _feature_matrix(values, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values.")
    return matrix


def _mean_scale(matrix: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0, ddof=1 if matrix.shape[0] > 1 else 0)
    return mean, np.maximum(scale, floor)


def _positive_float(value, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be positive and finite.")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed
