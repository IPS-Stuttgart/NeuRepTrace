"""Strict source-only random Fourier feature helpers.

This module builds random Fourier feature (RFF) maps for RBF-style kernels.  The
random projection is generated from the source feature width and seed, while the
``gamma='scale'`` heuristic is estimated from source rows only.  Held-out rows are
transformed with the frozen map and are never used to fit gamma or projection
parameters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_RFF_PROTOCOL = "strict_source_only_random_fourier_features"
SOURCE_RFF_CATEGORY = "1_strict_source_only"
DEFAULT_COMPONENTS = 256
DEFAULT_RANDOM_STATE = 13
DEFAULT_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class SourceRFFConfig:
    """Configuration for source-only random Fourier features."""

    n_components: int | str = DEFAULT_COMPONENTS
    gamma: float | str = "scale"
    random_state: int | None = DEFAULT_RANDOM_STATE
    standardize: bool = False
    epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True, slots=True)
class SourceRFFReference:
    """A fitted source-only RFF reference."""

    weights: np.ndarray
    phase: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    gamma: float
    config: SourceRFFConfig
    n_input_features: int


@dataclass(frozen=True, slots=True)
class SourceRFFResult:
    """RFF-transformed source/test rows and protocol metadata."""

    train_features: np.ndarray
    test_features: np.ndarray
    reference: SourceRFFReference
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments

def fit_source_rff_transform(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: SourceRFFConfig | Mapping[str, Any] | None = None,
) -> SourceRFFResult:
    """Fit an RFF map on source rows and transform source/test rows."""

    cfg = source_rff_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    test = _feature_matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError(f"source_features and test_features must have the same feature width: {source.shape[1]} != {test.shape[1]}.")
    reference = fit_source_rff_reference(source, config=cfg)
    train = apply_source_rff(source, reference)
    test_out = apply_source_rff(test, reference)
    metadata = _metadata(
        cfg,
        n_source_rows=source.shape[0],
        n_test_rows=test.shape[0],
        feature_dim=source.shape[1],
        n_components=reference.weights.shape[1],
        gamma=reference.gamma,
    )
    return SourceRFFResult(
        train_features=train.astype(np.float32, copy=False),
        test_features=test_out.astype(np.float32, copy=False),
        reference=reference,
        metadata=metadata,
    )


def fit_source_rff_reference(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    *,
    config: SourceRFFConfig | Mapping[str, Any] | None = None,
) -> SourceRFFReference:
    """Fit a source-only RFF reference.

    ``gamma='scale'`` follows the common RBF heuristic ``1 / (n_features * var)``
    using source rows only.
    """

    cfg = source_rff_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    mean = np.mean(source, axis=0) if cfg.standardize else np.zeros(source.shape[1], dtype=float)
    centered = source - mean
    if cfg.standardize:
        scale = np.std(centered, axis=0, ddof=1 if source.shape[0] > 1 else 0)
        scale = np.maximum(scale, cfg.epsilon)
        prepared = centered / scale
    else:
        scale = np.ones(source.shape[1], dtype=float)
        prepared = source
    gamma = _resolve_gamma(cfg.gamma, prepared, epsilon=cfg.epsilon)
    n_components = _positive_int(cfg.n_components, name="n_components")
    rng = np.random.default_rng(cfg.random_state)
    weights = rng.normal(0.0, np.sqrt(2.0 * gamma), size=(source.shape[1], n_components))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=n_components)
    return SourceRFFReference(
        weights=weights.astype(np.float32, copy=False),
        phase=phase.astype(np.float32, copy=False),
        mean=mean.astype(float, copy=False),
        scale=scale.astype(float, copy=False),
        gamma=float(gamma),
        config=cfg,
        n_input_features=int(source.shape[1]),
    )


def apply_source_rff(features: Sequence[Sequence[float]] | np.ndarray, reference: SourceRFFReference) -> np.ndarray:
    """Transform rows with a fitted source-only RFF reference."""

    matrix = _feature_matrix(features, name="features")
    if matrix.shape[1] != reference.n_input_features:
        raise ValueError(f"features width {matrix.shape[1]} does not match RFF reference width {reference.n_input_features}.")
    prepared = (matrix - reference.mean) / reference.scale
    projected = prepared @ reference.weights + reference.phase
    return np.sqrt(2.0 / reference.weights.shape[1]) * np.cos(projected)


def source_rff_config(
    *,
    n_components: int | str = DEFAULT_COMPONENTS,
    gamma: float | str = "scale",
    random_state: int | str | None = DEFAULT_RANDOM_STATE,
    standardize: bool | int | str = False,
    epsilon: float | str = DEFAULT_EPSILON,
) -> SourceRFFConfig:
    """Normalize source-RFF options."""

    return SourceRFFConfig(
        n_components=n_components,
        gamma=normalize_gamma(gamma),
        random_state=None if random_state in {None, "", "none", "None"} else _nonnegative_int(random_state, name="random_state"),
        standardize=_bool_value(standardize, name="standardize"),
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


def normalize_gamma(value: float | str) -> float | str:
    """Normalize RBF gamma aliases."""

    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_")
        if normalized in {"scale", "auto"}:
            return normalized
        value = normalized
    return _positive_float(value, name="gamma")


def _coerce_config(config: SourceRFFConfig | Mapping[str, Any]) -> SourceRFFConfig:
    if isinstance(config, SourceRFFConfig):
        return config
    return source_rff_config(**dict(config))


def _resolve_gamma(value: float | str, source: np.ndarray, *, epsilon: float) -> float:
    if isinstance(value, str):
        if value == "auto":
            return 1.0 / source.shape[1]
        if value == "scale":
            variance = float(np.var(source))
            return 1.0 / max(source.shape[1] * variance, float(epsilon))
        raise ValueError(f"Unknown gamma mode {value!r}.")
    return _positive_float(value, name="gamma")


def _metadata(cfg: SourceRFFConfig, *, n_source_rows: int, n_test_rows: int, feature_dim: int, n_components: int, gamma: float) -> dict[str, Any]:
    return {
        "source_rff": True,
        "source_rff_protocol": SOURCE_RFF_PROTOCOL,
        "source_rff_protocol_category": SOURCE_RFF_CATEGORY,
        "source_rff_uses_source_features": True,
        "source_rff_uses_source_labels": False,
        "source_rff_uses_test_features_for_fitting": False,
        "source_rff_uses_test_labels": False,
        "source_rff_valid_for_strict_source_only": True,
        "source_rff_valid_for_benchmark": True,
        "source_rff_n_source_rows": int(n_source_rows),
        "source_rff_n_test_rows": int(n_test_rows),
        "source_rff_feature_dim": int(feature_dim),
        "source_rff_n_components": int(n_components),
        "source_rff_requested_components": str(cfg.n_components),
        "source_rff_gamma": float(gamma),
        "source_rff_gamma_mode": str(cfg.gamma),
        "source_rff_random_state": "" if cfg.random_state is None else int(cfg.random_state),
        "source_rff_standardize": bool(cfg.standardize),
    }


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values.")
    return matrix


def _positive_int(value: int | str, *, name: str) -> int:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(parsed)


def _nonnegative_int(value: int | str, *, name: str) -> int:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(parsed)


def _positive_float(value: float | str, *, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed


def _bool_value(value: bool | int | str, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean value.")
