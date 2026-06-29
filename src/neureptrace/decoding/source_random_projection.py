"""Strict source-only random projection helpers.

This module creates a fold-local random projection matrix using only the source
feature dimension and a deterministic seed, then applies the fixed projection to
source and held-out rows.  It is a Protocol-1 preprocessing baseline: held-out
rows are transformed but never used to fit or tune the projection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_RANDOM_PROJECTION_PROTOCOL = "strict_source_only_random_projection"
SOURCE_RANDOM_PROJECTION_CATEGORY = "1_strict_source_only"
PROJECTION_DISTRIBUTIONS = ("gaussian", "sparse")
DEFAULT_COMPONENTS = 128
DEFAULT_RANDOM_STATE = 13


@dataclass(frozen=True, slots=True)
class SourceRandomProjectionConfig:
    """Configuration for a source-only random projection."""

    n_components: int | str = DEFAULT_COMPONENTS
    distribution: str = "gaussian"
    random_state: int | None = DEFAULT_RANDOM_STATE
    density: float | str = "auto"


@dataclass(frozen=True, slots=True)
class SourceRandomProjectionReference:
    """A fitted random projection reference."""

    projection: np.ndarray
    config: SourceRandomProjectionConfig
    n_input_features: int


@dataclass(frozen=True, slots=True)
class SourceRandomProjectionResult:
    """Projected source/test rows and protocol metadata."""

    train_features: np.ndarray
    test_features: np.ndarray
    reference: SourceRandomProjectionReference
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments

def fit_source_random_projection_transform(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: SourceRandomProjectionConfig | Mapping[str, Any] | None = None,
) -> SourceRandomProjectionResult:
    """Create a source-width random projection and transform source/test rows."""

    cfg = source_random_projection_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    test = _feature_matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError(f"source_features and test_features must have the same feature width: {source.shape[1]} != {test.shape[1]}.")
    reference = fit_source_random_projection_reference(source.shape[1], config=cfg)
    train = apply_source_random_projection(source, reference)
    test_out = apply_source_random_projection(test, reference)
    metadata = _metadata(cfg, n_source_rows=source.shape[0], n_test_rows=test.shape[0], feature_dim=source.shape[1], n_components=reference.projection.shape[1])
    return SourceRandomProjectionResult(
        train_features=train.astype(np.float32, copy=False),
        test_features=test_out.astype(np.float32, copy=False),
        reference=reference,
        metadata=metadata,
    )


def fit_source_random_projection_reference(
    n_features: int | str,
    *,
    config: SourceRandomProjectionConfig | Mapping[str, Any] | None = None,
) -> SourceRandomProjectionReference:
    """Create a random projection matrix for a source feature width."""

    cfg = source_random_projection_config() if config is None else _coerce_config(config)
    width = _positive_int(n_features, name="n_features")
    n_components = _resolve_components(cfg.n_components, n_features=width)
    rng = np.random.default_rng(cfg.random_state)
    if cfg.distribution == "gaussian":
        projection = rng.normal(0.0, 1.0 / np.sqrt(n_components), size=(width, n_components))
    elif cfg.distribution == "sparse":
        density = _resolve_density(cfg.density, n_features=width)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(width, n_components))
        keep = rng.random((width, n_components)) < density
        projection = signs * keep / np.sqrt(max(density * n_components, 1e-12))
    else:  # pragma: no cover - guarded by config normalization
        raise ValueError(f"Unhandled projection distribution {cfg.distribution!r}.")
    return SourceRandomProjectionReference(projection=projection.astype(np.float32, copy=False), config=cfg, n_input_features=width)


def apply_source_random_projection(features: Sequence[Sequence[float]] | np.ndarray, reference: SourceRandomProjectionReference) -> np.ndarray:
    """Apply a fitted source-width random projection."""

    matrix = _feature_matrix(features, name="features")
    if matrix.shape[1] != reference.n_input_features:
        raise ValueError(f"features width {matrix.shape[1]} does not match random projection width {reference.n_input_features}.")
    return (matrix @ reference.projection).astype(np.float32, copy=False)


def source_random_projection_config(
    *,
    n_components: int | str = DEFAULT_COMPONENTS,
    distribution: str | None = "gaussian",
    random_state: int | str | None = DEFAULT_RANDOM_STATE,
    density: float | str = "auto",
) -> SourceRandomProjectionConfig:
    """Normalize source-random-projection options."""

    return SourceRandomProjectionConfig(
        n_components=n_components,
        distribution=normalize_projection_distribution(distribution),
        random_state=None if random_state in {None, "", "none", "None"} else _nonnegative_int(random_state, name="random_state"),
        density=density if isinstance(density, str) and density.strip().lower() == "auto" else _unit_interval_open_float(density, name="density"),
    )


def normalize_projection_distribution(value: str | None) -> str:
    """Normalize projection distribution aliases."""

    normalized = "gaussian" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"normal": "gaussian", "dense": "gaussian", "achlioptas": "sparse"}.get(normalized, normalized)
    if normalized not in PROJECTION_DISTRIBUTIONS:
        raise ValueError(f"Unknown random projection distribution {value!r}.")
    return normalized


def _coerce_config(config: SourceRandomProjectionConfig | Mapping[str, Any]) -> SourceRandomProjectionConfig:
    if isinstance(config, SourceRandomProjectionConfig):
        return config
    return source_random_projection_config(**dict(config))


def _metadata(cfg: SourceRandomProjectionConfig, *, n_source_rows: int, n_test_rows: int, feature_dim: int, n_components: int) -> dict[str, Any]:
    return {
        "source_random_projection": True,
        "source_random_projection_protocol": SOURCE_RANDOM_PROJECTION_PROTOCOL,
        "source_random_projection_protocol_category": SOURCE_RANDOM_PROJECTION_CATEGORY,
        "source_random_projection_uses_source_feature_width": True,
        "source_random_projection_uses_source_values": False,
        "source_random_projection_uses_source_labels": False,
        "source_random_projection_uses_test_features_for_fitting": False,
        "source_random_projection_uses_test_labels": False,
        "source_random_projection_valid_for_strict_source_only": True,
        "source_random_projection_valid_for_benchmark": True,
        "source_random_projection_n_source_rows": int(n_source_rows),
        "source_random_projection_n_test_rows": int(n_test_rows),
        "source_random_projection_feature_dim": int(feature_dim),
        "source_random_projection_n_components": int(n_components),
        "source_random_projection_distribution": cfg.distribution,
        "source_random_projection_random_state": "" if cfg.random_state is None else int(cfg.random_state),
        "source_random_projection_density": str(cfg.density),
    }


def _resolve_components(value: int | str, *, n_features: int) -> int:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"all", "full"}:
            return int(n_features)
        requested = float(text)
    else:
        requested = float(value)
    if not np.isfinite(requested) or requested % 1.0 != 0.0 or requested < 1:
        raise ValueError("n_components must be a positive integer, 'all', or 'full'.")
    return int(requested)


def _resolve_density(value: float | str, *, n_features: int) -> float:
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "auto":
            return min(1.0, 1.0 / np.sqrt(n_features))
        value = text
    return _unit_interval_open_float(value, name="density")


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


def _unit_interval_open_float(value: float | str, *, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be in (0, 1].")
    return parsed
