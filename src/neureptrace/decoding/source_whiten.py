"""Strict source-only whitening transforms.

This module fits feature-centering and whitening matrices from source rows only and
then applies the frozen transform to source and held-out rows.  It provides a
small fold-local preprocessing baseline for cross-subject decoding while keeping
all held-out data out of the fitting path.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_WHITEN_PROTOCOL = "strict_source_only_whitening"
SOURCE_WHITEN_CATEGORY = "1_strict_source_only"
WHITEN_METHODS = ("pca", "zca")
DEFAULT_REGULARIZATION = 1e-6


@dataclass(frozen=True, slots=True)
class SourceWhitenConfig:
    """Configuration for source-fitted whitening."""

    method: str = "pca"
    n_components: int | str | None = "all"
    center: bool = True
    regularization: float = DEFAULT_REGULARIZATION

    def __post_init__(self) -> None:
        """Normalize and validate direct dataclass construction."""

        object.__setattr__(self, "method", normalize_whiten_method(self.method))
        object.__setattr__(self, "n_components", _normalize_n_components_request(self.n_components))
        object.__setattr__(self, "center", _bool_config(self.center, name="center"))
        object.__setattr__(self, "regularization", _nonnegative_float(self.regularization, name="regularization"))


@dataclass(frozen=True, slots=True)
class SourceWhitenTransform:
    """Frozen source-fitted whitening transform."""

    mean: np.ndarray
    components: np.ndarray
    whitening: np.ndarray
    eigenvalues: np.ndarray
    method: str
    n_source_rows: int
    feature_dim: int


@dataclass(frozen=True, slots=True)
class SourceWhitenResult:
    """Whitened source/test features and provenance metadata."""

    train_features: np.ndarray
    test_features: np.ndarray
    transform: SourceWhitenTransform
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments)

def fit_source_whiten(
    *,
    source_features: Iterable[Iterable[float]] | np.ndarray,
    test_features: Iterable[Iterable[float]] | np.ndarray,
    config: SourceWhitenConfig | Mapping[str, Any] | None = None,
) -> SourceWhitenResult:
    """Fit a whitening transform from source rows and apply it to two matrices."""

    cfg = source_whiten_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    test = _feature_matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError(
            "source_features and test_features must have the same feature width: "
            f"{source.shape[1]} != {test.shape[1]}."
        )
    transform = fit_source_whiten_transform(source, config=cfg)
    train = apply_source_whiten(source, transform)
    test_out = apply_source_whiten(test, transform)
    metadata = _metadata(cfg, n_source_rows=source.shape[0], n_test_rows=test.shape[0], feature_dim=source.shape[1], output_dim=train.shape[1])
    return SourceWhitenResult(
        train_features=train.astype(np.float32, copy=False),
        test_features=test_out.astype(np.float32, copy=False),
        transform=transform,
        metadata=metadata,
    )


def source_whiten_config(
    *,
    method: str | None = "pca",
    n_components: int | str | None = "all",
    center: bool | str | int | float = True,
    regularization: float | str = DEFAULT_REGULARIZATION,
) -> SourceWhitenConfig:
    """Normalize public source-whitening options."""

    return SourceWhitenConfig(
        method=method,
        n_components=n_components,
        center=center,
        regularization=regularization,
    )


def normalize_whiten_method(value: str | None) -> str:
    """Normalize whitening method aliases."""

    normalized = "pca" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"pcawhiten": "pca", "pca_whiten": "pca", "zcawhiten": "zca", "zca_whiten": "zca"}.get(normalized, normalized)
    if normalized not in WHITEN_METHODS:
        raise ValueError(f"Unknown whitening method {value!r}. Available values: {', '.join(WHITEN_METHODS)}.")
    return normalized


def fit_source_whiten_transform(
    source_features: Iterable[Iterable[float]] | np.ndarray,
    *,
    config: SourceWhitenConfig | Mapping[str, Any] | None = None,
) -> SourceWhitenTransform:
    """Estimate a whitening transform from source rows only."""

    cfg = source_whiten_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    mean = np.mean(source, axis=0) if cfg.center else np.zeros(source.shape[1], dtype=float)
    centered = source - mean
    covariance = _covariance(centered)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    vectors = _canonicalize_component_signs(vectors[:, order].T)
    n_components = _effective_components(cfg.n_components, max_components=source.shape[1])
    if cfg.method == "pca":
        components = vectors[:n_components]
        whitening = components.T / np.sqrt(values[:n_components] + cfg.regularization)
    elif cfg.method == "zca":
        if n_components != source.shape[1]:
            raise ValueError("ZCA whitening requires n_components='all' or the full feature dimension.")
        components = vectors
        whitening = vectors.T @ np.diag(1.0 / np.sqrt(values + cfg.regularization)) @ vectors
    else:  # pragma: no cover - guarded by normalization
        raise ValueError(f"Unhandled whitening method {cfg.method!r}.")
    return SourceWhitenTransform(
        mean=mean.astype(float, copy=False),
        components=components.astype(np.float32, copy=False),
        whitening=whitening.astype(np.float32, copy=False),
        eigenvalues=values[: components.shape[0]].astype(float, copy=False),
        method=cfg.method,
        n_source_rows=int(source.shape[0]),
        feature_dim=int(source.shape[1]),
    )


def apply_source_whiten(features: Iterable[Iterable[float]] | np.ndarray, transform: SourceWhitenTransform) -> np.ndarray:
    """Apply a frozen source-fitted whitening transform."""

    matrix = _feature_matrix(features, name="features")
    if matrix.shape[1] != transform.feature_dim:
        raise ValueError(f"features width {matrix.shape[1]} does not match transform width {transform.feature_dim}.")
    centered = matrix - transform.mean
    return centered @ transform.whitening


def _coerce_config(config: SourceWhitenConfig | Mapping[str, Any]) -> SourceWhitenConfig:
    if isinstance(config, SourceWhitenConfig):
        return source_whiten_config(
            method=config.method,
            n_components=config.n_components,
            center=config.center,
            regularization=config.regularization,
        )
    if not isinstance(config, Mapping):
        raise ValueError("source whitening config must be a mapping or SourceWhitenConfig.")
    raw = dict(config)
    allowed = {"method", "n_components", "center", "regularization"}
    unknown = sorted(str(key) for key in raw if key not in allowed)
    if unknown:
        raise ValueError(f"Unknown source whitening config option(s): {', '.join(unknown)}. Available options: {', '.join(sorted(allowed))}.")
    return source_whiten_config(**raw)


def _covariance(centered: np.ndarray) -> np.ndarray:
    if centered.shape[0] <= 1:
        return np.zeros((centered.shape[1], centered.shape[1]), dtype=float)
    return centered.T @ centered / float(centered.shape[0] - 1)


def _normalize_n_components_request(value: int | str | None) -> int | str | None:
    message = "n_components must be a positive integer or 'all'."
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.ndim != 0 or np.issubdtype(value.dtype, np.bool_):
            raise ValueError(message)
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "all", "full", "none"}:
            return "all"
        value = text
    elif isinstance(value, (bool, np.bool_, list, tuple, dict, set)):
        raise ValueError(message)
    try:
        requested = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(requested) or requested % 1.0 != 0.0 or requested < 1:
        raise ValueError(message)
    return int(requested)


def _effective_components(value: int | str | None, *, max_components: int) -> int:
    requested = _normalize_n_components_request(value)
    if requested is None or isinstance(requested, str):
        return int(max_components)
    return min(int(requested), int(max_components))


def _canonicalize_component_signs(components: np.ndarray) -> np.ndarray:
    output = np.asarray(components, dtype=float).copy()
    for row in range(output.shape[0]):
        pivot = int(np.argmax(np.abs(output[row])))
        if output[row, pivot] < 0.0:
            output[row] *= -1.0
    return output


def _metadata(cfg: SourceWhitenConfig, *, n_source_rows: int, n_test_rows: int, feature_dim: int, output_dim: int) -> dict[str, Any]:
    return {
        "source_whiten": True,
        "source_whiten_protocol": SOURCE_WHITEN_PROTOCOL,
        "source_whiten_protocol_category": SOURCE_WHITEN_CATEGORY,
        "source_whiten_uses_source_features": True,
        "source_whiten_uses_test_features_for_fitting": False,
        "source_whiten_uses_test_labels": False,
        "source_whiten_valid_for_strict_source_only": True,
        "source_whiten_valid_for_benchmark": True,
        "source_whiten_n_source_rows": int(n_source_rows),
        "source_whiten_n_test_rows": int(n_test_rows),
        "source_whiten_feature_dim": int(feature_dim),
        "source_whiten_output_dim": int(output_dim),
        "source_whiten_method": cfg.method,
        "source_whiten_requested_components": str(cfg.n_components),
        "source_whiten_center": bool(cfg.center),
        "source_whiten_regularization": float(cfg.regularization),
    }


def _materialize_one_pass_iterables(value: object) -> object:
    """Materialize nested one-pass iterables before NumPy consumes them."""

    if isinstance(value, np.ndarray):
        if value.dtype != object:
            return value
        return _materialize_one_pass_iterables(value.tolist())
    if isinstance(value, (str, bytes)):
        return value
    if not isinstance(value, Iterable):
        return value
    return [_materialize_one_pass_iterables(item) for item in value]


def _contains_boolean_value(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        if value.dtype == np.bool_:
            return True
        if value.dtype == object:
            return any(_contains_boolean_value(item) for item in value.ravel(order="C"))
        return False
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, np.generic):
        return isinstance(value.item(), (bool, np.bool_))
    if isinstance(value, Iterable):
        return any(_contains_boolean_value(item) for item in value)
    return False


def _feature_matrix(values: Iterable[Iterable[float]] | np.ndarray, *, name: str) -> np.ndarray:
    materialized = _materialize_one_pass_iterables(values)
    if _contains_boolean_value(materialized):
        raise ValueError(f"{name} must contain numeric feature values, not boolean flags.")
    matrix = np.asarray(materialized, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _bool_config(value: bool | str | int | float, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "f", "no", "n", "off"}:
            return False
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(float(value)) and float(value) in {0.0, 1.0}:
        return bool(value)
    raise ValueError(f"{name} must be a boolean value.")


def _nonnegative_float(value: float | str, *, name: str) -> float:
    message = f"{name} must be non-negative and finite."
    if isinstance(value, np.ndarray):
        if value.ndim != 0 or np.issubdtype(value.dtype, np.bool_):
            raise ValueError(message)
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bool, np.bool_, list, tuple, dict, set)):
        raise ValueError(message)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(message)
    return parsed
