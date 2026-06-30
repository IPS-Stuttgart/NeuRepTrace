"""Feature normalization protocols for source and held-out target domains.

This module provides small, auditable normalization transforms for cross-subject
M/EEG decoding.  The strict source-only mode estimates all statistics from source
rows.  The adaptive modes estimate label-free target statistics and are therefore
Category 2 / unlabeled target-adaptive.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

NORMALIZATION_MODES = ("source_only", "pooled", "target", "domain_wise")
ADAPTIVE_NORMALIZATION_PROTOCOL = "feature_statistic_normalization"
CATEGORY_SOURCE_ONLY = "1_strict_source_only"
CATEGORY_UNLABELED_TARGET = "2_unlabeled_target_adaptive"
DEFAULT_EPSILON = 1e-8


@dataclass(frozen=True, slots=True)
class FeatureNormalizationStats:
    """Mean and scale estimates for one feature domain."""

    mean: np.ndarray
    scale: np.ndarray
    n_rows: int


@dataclass(frozen=True, slots=True)
class AdaptiveNormalizationConfig:
    """Configuration for fold-local feature normalization."""

    mode: str = "domain_wise"
    center: bool = True
    scale: bool = True
    robust: bool = False
    epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True, slots=True)
class AdaptiveNormalizationResult:
    """Normalized train/test features and protocol metadata."""

    train_features: np.ndarray
    test_features: np.ndarray
    source_stats: FeatureNormalizationStats
    target_stats: FeatureNormalizationStats | None
    pooled_stats: FeatureNormalizationStats | None
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments

def fit_adaptive_feature_normalization(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    target_features: Sequence[Sequence[float]] | np.ndarray,
    config: AdaptiveNormalizationConfig | Mapping[str, Any] | None = None,
    target_adaptation_features: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> AdaptiveNormalizationResult:
    """Normalize source and target feature rows under an explicit protocol.

    Parameters
    ----------
    source_features:
        Source train rows.
    target_features:
        Held-out target rows to transform.
    config:
        Normalization settings.  A mapping is normalized through
        :func:`adaptive_normalization_config`.
    target_adaptation_features:
        Optional unlabeled target rows used to estimate target statistics.  When
        omitted, ``target_features`` are used as the adaptation batch, which is a
        transductive Category-2 setting for adaptive modes.

    Returns
    -------
    AdaptiveNormalizationResult
        Normalized train/test features and provenance metadata.
    """

    cfg = adaptive_normalization_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    target = _feature_matrix(target_features, name="target_features")
    if source.shape[1] != target.shape[1]:
        raise ValueError(f"source_features and target_features must have the same feature width: {source.shape[1]} != {target.shape[1]}.")
    target_fit = target if target_adaptation_features is None else _feature_matrix(target_adaptation_features, name="target_adaptation_features")
    if target_fit.shape[1] != source.shape[1]:
        raise ValueError(f"target_adaptation_features and source_features must have the same feature width: {target_fit.shape[1]} != {source.shape[1]}.")

    source_stats = estimate_feature_normalization_stats(source, center=cfg.center, scale=cfg.scale, robust=cfg.robust, epsilon=cfg.epsilon)
    target_stats = None
    pooled_stats = None

    if cfg.mode == "source_only":
        train = apply_feature_normalization(source, source_stats)
        test = apply_feature_normalization(target, source_stats)
    elif cfg.mode == "pooled":
        pooled_stats = estimate_feature_normalization_stats(np.vstack([source, target_fit]), center=cfg.center, scale=cfg.scale, robust=cfg.robust, epsilon=cfg.epsilon)
        train = apply_feature_normalization(source, pooled_stats)
        test = apply_feature_normalization(target, pooled_stats)
    else:
        target_stats = estimate_feature_normalization_stats(target_fit, center=cfg.center, scale=cfg.scale, robust=cfg.robust, epsilon=cfg.epsilon)
        if cfg.mode == "target":
            train = apply_feature_normalization(source, target_stats)
            test = apply_feature_normalization(target, target_stats)
        elif cfg.mode == "domain_wise":
            train = apply_feature_normalization(source, source_stats)
            test = apply_feature_normalization(target, target_stats)
        else:  # pragma: no cover - guarded by config normalization
            raise ValueError(f"Unhandled normalization mode {cfg.mode!r}.")

    metadata = _metadata(
        cfg,
        n_source_rows=source.shape[0],
        n_target_rows=target.shape[0],
        n_target_fit_rows=target_fit.shape[0],
        feature_dim=source.shape[1],
        transductive=target_adaptation_features is None and cfg.mode != "source_only",
    )
    return AdaptiveNormalizationResult(
        train_features=train.astype(np.float32, copy=False),
        test_features=test.astype(np.float32, copy=False),
        source_stats=source_stats,
        target_stats=target_stats,
        pooled_stats=pooled_stats,
        metadata=metadata,
    )


def adaptive_normalization_config(
    *,
    mode: str | None = "domain_wise",
    center: Any = True,
    scale: Any = True,
    robust: Any = False,
    epsilon: float | str = DEFAULT_EPSILON,
) -> AdaptiveNormalizationConfig:
    """Normalize public options for adaptive feature normalization."""

    return AdaptiveNormalizationConfig(
        mode=normalize_adaptive_normalization_mode(mode),
        center=_bool_config(center, name="center"),
        scale=_bool_config(scale, name="scale"),
        robust=_bool_config(robust, name="robust"),
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


def _bool_config(value: Any, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean value.")


def normalize_adaptive_normalization_mode(mode: str | None) -> str:
    """Normalize mode aliases."""

    normalized = "domain_wise" if mode is None else str(mode).strip().lower().replace("-", "_")
    normalized = {
        "source": "source_only",
        "source_only": "source_only",
        "train": "source_only",
        "train_only": "source_only",
        "strict_source_only": "source_only",
        "source_target": "pooled",
        "source_plus_target": "pooled",
        "all": "pooled",
        "all_unlabeled": "pooled",
        "target_stats": "target",
        "target_only": "target",
        "target_adaptive": "target",
        "per_domain": "domain_wise",
        "domainwise": "domain_wise",
        "domain_wise": "domain_wise",
        "adaptive_batch_norm": "domain_wise",
        "adabn": "domain_wise",
    }.get(normalized, normalized)
    if normalized not in NORMALIZATION_MODES:
        raise ValueError(f"Unknown adaptive normalization mode {mode!r}. Available modes: {', '.join(NORMALIZATION_MODES)}.")
    return normalized


def estimate_feature_normalization_stats(
    features: Sequence[Sequence[float]] | np.ndarray,
    *,
    center: bool = True,
    scale: bool = True,
    robust: bool = False,
    epsilon: float = DEFAULT_EPSILON,
) -> FeatureNormalizationStats:
    """Estimate feature-wise normalization statistics."""

    matrix = _feature_matrix(features, name="features")
    eps = _positive_float(epsilon, name="epsilon")
    if center:
        mean = np.median(matrix, axis=0) if robust else np.mean(matrix, axis=0)
    else:
        mean = np.zeros(matrix.shape[1], dtype=float)
    centered = matrix - mean
    if scale:
        if robust:
            q75 = np.percentile(matrix, 75.0, axis=0)
            q25 = np.percentile(matrix, 25.0, axis=0)
            scale_vector = (q75 - q25) / 1.349
        else:
            scale_vector = np.std(centered, axis=0, ddof=1 if matrix.shape[0] > 1 else 0)
        scale_vector = np.maximum(np.asarray(scale_vector, dtype=float), eps)
    else:
        scale_vector = np.ones(matrix.shape[1], dtype=float)
    return FeatureNormalizationStats(mean=np.asarray(mean, dtype=float), scale=scale_vector, n_rows=int(matrix.shape[0]))


def apply_feature_normalization(features: Sequence[Sequence[float]] | np.ndarray, stats: FeatureNormalizationStats) -> np.ndarray:
    """Apply precomputed feature-normalization statistics."""

    matrix = _feature_matrix(features, name="features")
    if matrix.shape[1] != stats.mean.shape[0] or matrix.shape[1] != stats.scale.shape[0]:
        raise ValueError("features width must match stats mean/scale length.")
    return (matrix - stats.mean) / stats.scale


def _coerce_config(config: AdaptiveNormalizationConfig | Mapping[str, Any]) -> AdaptiveNormalizationConfig:
    if isinstance(config, AdaptiveNormalizationConfig):
        return config
    return adaptive_normalization_config(**dict(config))


def _metadata(
    cfg: AdaptiveNormalizationConfig,
    *,
    n_source_rows: int,
    n_target_rows: int,
    n_target_fit_rows: int,
    feature_dim: int,
    transductive: bool,
) -> dict[str, Any]:
    adaptive = cfg.mode != "source_only"
    return {
        "adaptive_feature_normalization": True,
        "adaptive_feature_normalization_protocol": ADAPTIVE_NORMALIZATION_PROTOCOL,
        "adaptive_feature_normalization_protocol_category": CATEGORY_UNLABELED_TARGET if adaptive else CATEGORY_SOURCE_ONLY,
        "adaptive_feature_normalization_mode": cfg.mode,
        "adaptive_feature_normalization_uses_source_features": True,
        "adaptive_feature_normalization_uses_target_features": bool(adaptive),
        "adaptive_feature_normalization_uses_target_labels": False,
        "adaptive_feature_normalization_valid_for_strict_source_only": not adaptive,
        "adaptive_feature_normalization_valid_for_unlabeled_target_adaptation": True,
        "adaptive_feature_normalization_valid_for_benchmark": not adaptive,
        "adaptive_feature_normalization_transductive": bool(transductive),
        "adaptive_feature_normalization_n_source_rows": int(n_source_rows),
        "adaptive_feature_normalization_n_target_rows": int(n_target_rows),
        "adaptive_feature_normalization_n_target_fit_rows": int(n_target_fit_rows),
        "adaptive_feature_normalization_feature_dim": int(feature_dim),
        "adaptive_feature_normalization_center": bool(cfg.center),
        "adaptive_feature_normalization_scale": bool(cfg.scale),
        "adaptive_feature_normalization_robust": bool(cfg.robust),
        "adaptive_feature_normalization_epsilon": float(cfg.epsilon),
    }


def _feature_matrix(features: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    if matrix.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")
    if matrix.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature column.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _positive_float(value: float | str, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite float.") from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be a positive finite float.")
    return parsed
