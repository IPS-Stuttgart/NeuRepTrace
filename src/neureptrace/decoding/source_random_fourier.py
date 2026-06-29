"""Strict source-only random Fourier feature helpers.

This module creates random Fourier features for RBF-style nonlinear probes.  The
random basis is determined from the source feature width, a source-only gamma
setting, and a deterministic seed.  Held-out rows are transformed with the fixed
basis but are never used to fit or tune the transform.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_RANDOM_FOURIER_PROTOCOL = "strict_source_only_random_fourier_features"
SOURCE_RANDOM_FOURIER_CATEGORY = "1_strict_source_only"
DEFAULT_COMPONENTS = 256
DEFAULT_RANDOM_STATE = 13
DEFAULT_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class SourceRandomFourierConfig:
    """Configuration for source-only random Fourier features."""

    n_components: int | str = DEFAULT_COMPONENTS
    gamma: float | str = "auto"
    random_state: int | None = DEFAULT_RANDOM_STATE
    include_original: bool = False
    epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True, slots=True)
class SourceRandomFourierReference:
    """A fitted random Fourier feature reference."""

    weights: np.ndarray
    offsets: np.ndarray
    gamma: float
    config: SourceRandomFourierConfig
    n_input_features: int


@dataclass(frozen=True, slots=True)
class SourceRandomFourierResult:
    """Transformed source/test rows and protocol metadata."""

    train_features: np.ndarray
    test_features: np.ndarray
    reference: SourceRandomFourierReference
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments
def fit_source_random_fourier_transform(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: SourceRandomFourierConfig | Mapping[str, Any] | None = None,
) -> SourceRandomFourierResult:
    """Fit a source-only random Fourier reference and transform source/test rows."""

    cfg = source_random_fourier_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    test = _feature_matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError(f"source_features and test_features must have the same feature width: {source.shape[1]} != {test.shape[1]}.")
    reference = fit_source_random_fourier_reference(source, config=cfg)
    train = apply_source_random_fourier(source, reference)
    test_out = apply_source_random_fourier(test, reference)
    metadata = _metadata(
        cfg,
        n_source_rows=source.shape[0],
        n_test_rows=test.shape[0],
        feature_dim=source.shape[1],
        n_output_features=train.shape[1],
        gamma=reference.gamma,
    )
    return SourceRandomFourierResult(
        train_features=train.astype(np.float32, copy=False),
        test_features=test_out.astype(np.float32, copy=False),
        reference=reference,
        metadata=metadata,
    )


def fit_source_random_fourier_reference(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    *,
    config: SourceRandomFourierConfig | Mapping[str, Any] | None = None,
) -> SourceRandomFourierReference:
    """Fit a random Fourier reference using source rows only."""

    cfg = source_random_fourier_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    components = _resolve_components(cfg.n_components)
    gamma = _resolve_gamma(cfg.gamma, source, epsilon=cfg.epsilon)
    rng = np.random.default_rng(cfg.random_state)
    weights = rng.normal(0.0, np.sqrt(2.0 * gamma), size=(source.shape[1], components))
    offsets = rng.uniform(0.0, 2.0 * np.pi, size=components)
    return SourceRandomFourierReference(
        weights=weights.astype(np.float32, copy=False),
        offsets=offsets.astype(np.float32, copy=False),
        gamma=float(gamma),
        config=cfg,
        n_input_features=int(source.shape[1]),
    )


def apply_source_random_fourier(features: Sequence[Sequence[float]] | np.ndarray, reference: SourceRandomFourierReference) -> np.ndarray:
    """Apply a fitted source-only random Fourier transform."""

    matrix = _feature_matrix(features, name="features")
    if matrix.shape[1] != reference.n_input_features:
        raise ValueError(f"features width {matrix.shape[1]} does not match random Fourier reference width {reference.n_input_features}.")
    projection = matrix @ reference.weights + reference.offsets
    transformed = np.sqrt(2.0 / reference.weights.shape[1]) * np.cos(projection)
    if reference.config.include_original:
        transformed = np.hstack([matrix, transformed])
    return transformed.astype(np.float32, copy=False)


def source_random_fourier_config(
    *,
    n_components: int | str = DEFAULT_COMPONENTS,
    gamma: float | str = "auto",
    random_state: int | str | None = DEFAULT_RANDOM_STATE,
    include_original: bool | int | str = False,
    epsilon: float | str = DEFAULT_EPSILON,
) -> SourceRandomFourierConfig:
    """Normalize source random Fourier feature options."""

    return SourceRandomFourierConfig(
        n_components=n_components,
        gamma=gamma if isinstance(gamma, str) and gamma.strip().lower() == "auto" else _positive_float(gamma, name="gamma"),
        random_state=None if random_state in {None, "", "none", "None"} else _nonnegative_int(random_state, name="random_state"),
        include_original=_bool_value(include_original, name="include_original"),
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


def source_auto_rbf_gamma(source_features: Sequence[Sequence[float]] | np.ndarray, *, epsilon: float = DEFAULT_EPSILON) -> float:
    """Estimate an RBF gamma from source rows by the median-distance heuristic."""

    source = _feature_matrix(source_features, name="source_features")
    squared = _squared_euclidean(source, source)
    positive = squared[squared > float(epsilon)]
    if positive.size == 0:
        return 1.0
    return float(1.0 / (2.0 * max(float(np.median(positive)), float(epsilon))))


def _coerce_config(config: SourceRandomFourierConfig | Mapping[str, Any]) -> SourceRandomFourierConfig:
    if isinstance(config, SourceRandomFourierConfig):
        return config
    return source_random_fourier_config(**dict(config))


def _metadata(cfg: SourceRandomFourierConfig, *, n_source_rows: int, n_test_rows: int, feature_dim: int, n_output_features: int, gamma: float) -> dict[str, Any]:
    return {
        "source_random_fourier": True,
        "source_random_fourier_protocol": SOURCE_RANDOM_FOURIER_PROTOCOL,
        "source_random_fourier_protocol_category": SOURCE_RANDOM_FOURIER_CATEGORY,
        "source_random_fourier_uses_source_features": True,
        "source_random_fourier_uses_source_labels": False,
        "source_random_fourier_uses_test_features_for_fitting": False,
        "source_random_fourier_uses_test_labels": False,
        "source_random_fourier_valid_for_strict_source_only": True,
        "source_random_fourier_valid_for_benchmark": True,
        "source_random_fourier_n_source_rows": int(n_source_rows),
        "source_random_fourier_n_test_rows": int(n_test_rows),
        "source_random_fourier_feature_dim": int(feature_dim),
        "source_random_fourier_n_components": int(_resolve_components(cfg.n_components)),
        "source_random_fourier_n_output_features": int(n_output_features),
        "source_random_fourier_gamma": float(gamma),
        "source_random_fourier_gamma_mode": "auto" if isinstance(cfg.gamma, str) and cfg.gamma == "auto" else "fixed",
        "source_random_fourier_random_state": "" if cfg.random_state is None else int(cfg.random_state),
        "source_random_fourier_include_original": bool(cfg.include_original),
    }


def _resolve_gamma(value: float | str, source: np.ndarray, *, epsilon: float) -> float:
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "auto":
            return source_auto_rbf_gamma(source, epsilon=epsilon)
        value = text
    return _positive_float(value, name="gamma")


def _resolve_components(value: int | str) -> int:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "default"}:
            requested = DEFAULT_COMPONENTS
        else:
            requested = float(text)
    else:
        requested = float(value)
    if not np.isfinite(requested) or requested % 1.0 != 0.0 or requested < 1:
        raise ValueError("n_components must be a positive integer.")
    return int(requested)


def _squared_euclidean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.maximum(np.sum(left * left, axis=1, keepdims=True) + np.sum(right * right, axis=1, keepdims=True).T - 2.0 * (left @ right.T), 0.0)


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values.")
    return matrix


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
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean value.")
