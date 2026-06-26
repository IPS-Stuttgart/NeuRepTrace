"""Category-2 source-target subspace adaptation utilities.

The helpers in this module implement a small dependency-light Transfer Component
Analysis style projection for cross-subject M/EEG features.  The projection is fit
from source features and unlabeled target features; optional source labels can be
used only to balance the source side of the marginal domain discrepancy.  Held-out
target labels are intentionally absent from the public API.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.linalg import eigh

SUBSPACE_ADAPTATION_PROTOCOL = "unlabeled_target_subspace_adaptation"
SUBSPACE_ADAPTATION_CATEGORY = "2_unlabeled_target_adaptive"
SUBSPACE_METHODS = ("tca", "balanced_tca")
DEFAULT_SUBSPACE_METHOD = "tca"
DEFAULT_SUBSPACE_COMPONENTS = 16
DEFAULT_SUBSPACE_REGULARIZATION = 1e-3
DEFAULT_SUBSPACE_EIGEN_RIDGE = 1e-6
MIN_SCALE = 1e-12


@dataclass(frozen=True, slots=True)
class SubspaceAdaptationConfig:
    """Configuration for Category-2 TCA-style feature projection."""

    method: str = DEFAULT_SUBSPACE_METHOD
    n_components: int | str = DEFAULT_SUBSPACE_COMPONENTS
    regularization: float = DEFAULT_SUBSPACE_REGULARIZATION
    eigen_ridge: float = DEFAULT_SUBSPACE_EIGEN_RIDGE
    standardize: bool = True
    class_balance_source: bool = False
    normalize_latent: bool = False


@dataclass(frozen=True, slots=True)
class SubspaceAdaptationResult:
    """Source/target features projected into a target-adaptive latent space."""

    source_features: np.ndarray
    target_features: np.ndarray
    projection: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    eigenvalues: np.ndarray
    source_weights: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments,too-many-locals

def fit_subspace_adaptation(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    target_features: Sequence[Sequence[float]] | np.ndarray,
    *,
    source_labels: Sequence[Any] | np.ndarray | None = None,
    config: SubspaceAdaptationConfig | dict[str, Any] | None = None,
    method: str | None = None,
    n_components: int | str | None = None,
    regularization: float | str | None = None,
    eigen_ridge: float | str | None = None,
    standardize: bool | None = None,
    class_balance_source: bool | None = None,
    normalize_latent: bool | None = None,
) -> SubspaceAdaptationResult:
    """Fit a TCA-style subspace using source and unlabeled target features.

    Parameters
    ----------
    source_features, target_features:
        Source and held-out target feature matrices with the same feature width.
        Target rows are used without labels to fit the shared projection.
    source_labels:
        Optional source labels.  They are used only when source class balancing is
        requested, so source classes contribute equal mass to the marginal
        source-target discrepancy.
    config, method, n_components, regularization, eigen_ridge, standardize,
    class_balance_source, normalize_latent:
        Configuration values.  Explicit keyword values override ``config``.

    Returns
    -------
    SubspaceAdaptationResult
        Projected source and target features plus projection/provenance fields.

    Notes
    -----
    This is a Category-2 protocol.  It uses ``X_s`` and unlabeled ``X_t``; it may
    use ``y_s`` for source-side class balancing; it never accepts target labels.
    """

    cfg = _resolve_config(
        config,
        method=method,
        n_components=n_components,
        regularization=regularization,
        eigen_ridge=eigen_ridge,
        standardize=standardize,
        class_balance_source=class_balance_source,
        normalize_latent=normalize_latent,
    )
    source = _feature_matrix(source_features, name="source_features")
    target = _feature_matrix(target_features, name="target_features")
    if source.shape[1] != target.shape[1]:
        raise ValueError(f"source_features and target_features must have the same feature width: {source.shape[1]} != {target.shape[1]}.")
    if cfg.class_balance_source and source_labels is None:
        raise ValueError("class_balance_source=True requires source_labels.")
    labels = None if source_labels is None else _object_vector(source_labels, expected_length=source.shape[0], name="source_labels")

    joint = np.vstack([source, target]).astype(float, copy=False)
    mean = np.mean(joint, axis=0) if cfg.standardize else np.zeros(joint.shape[1], dtype=float)
    centered = joint - mean
    scale = np.std(centered, axis=0, ddof=1 if joint.shape[0] > 1 else 0) if cfg.standardize else np.ones(joint.shape[1], dtype=float)
    scale = np.maximum(scale, MIN_SCALE)
    z = centered / scale

    source_weights = _source_weights(source.shape[0], labels=labels, class_balance=cfg.class_balance_source)
    target_weights = np.full(target.shape[0], 1.0 / float(target.shape[0]), dtype=float)
    domain_vector = np.concatenate([source_weights, -target_weights])
    mmd_matrix = np.outer(domain_vector, domain_vector)
    centering = np.eye(joint.shape[0], dtype=float) - np.full((joint.shape[0], joint.shape[0]), 1.0 / float(joint.shape[0]))

    feature_dim = z.shape[1]
    n_components_resolved = _effective_components(cfg.n_components, n_samples=z.shape[0], n_features=feature_dim)
    a_matrix = z.T @ mmd_matrix @ z + cfg.regularization * np.eye(feature_dim, dtype=float)
    b_matrix = z.T @ centering @ z + cfg.eigen_ridge * np.eye(feature_dim, dtype=float)
    values, vectors = eigh(a_matrix, b_matrix, check_finite=True)
    order = np.argsort(values)
    selected = order[:n_components_resolved]
    projection = _canonicalize_projection(vectors[:, selected])
    latent = z @ projection
    if cfg.normalize_latent:
        latent_scale = np.maximum(np.std(latent, axis=0, ddof=1 if latent.shape[0] > 1 else 0), MIN_SCALE)
        latent = latent / latent_scale
        projection = projection / latent_scale.reshape(1, -1)
    source_latent = latent[: source.shape[0]]
    target_latent = latent[source.shape[0] :]

    raw_gap = _weighted_mean_gap((source - mean) / scale, (target - mean) / scale, source_weights=source_weights, target_weights=target_weights)
    latent_gap = _weighted_mean_gap(source_latent, target_latent, source_weights=source_weights, target_weights=target_weights)
    metadata = _metadata(
        cfg=cfg,
        n_source_rows=source.shape[0],
        n_target_rows=target.shape[0],
        feature_dim=feature_dim,
        n_components=n_components_resolved,
        source_labels_used=labels is not None and cfg.class_balance_source,
        raw_gap=raw_gap,
        latent_gap=latent_gap,
        eigenvalues=values[selected],
    )
    return SubspaceAdaptationResult(
        source_features=source_latent.astype(np.float32, copy=False),
        target_features=target_latent.astype(np.float32, copy=False),
        projection=projection.astype(np.float32, copy=False),
        feature_mean=mean.astype(np.float32, copy=False),
        feature_scale=scale.astype(np.float32, copy=False),
        eigenvalues=np.asarray(values[selected], dtype=float),
        source_weights=source_weights.astype(float, copy=False),
        metadata=metadata,
    )


def transform_subspace_features(features: Sequence[Sequence[float]] | np.ndarray, result: SubspaceAdaptationResult) -> np.ndarray:
    """Transform new rows with an already fitted subspace adaptation result."""

    matrix = _feature_matrix(features, name="features")
    if matrix.shape[1] != result.projection.shape[0]:
        raise ValueError(f"features width {matrix.shape[1]} does not match projection width {result.projection.shape[0]}.")
    return (((matrix - result.feature_mean) / result.feature_scale) @ result.projection).astype(np.float32, copy=False)


def subspace_adaptation_config(
    *,
    method: str | None = DEFAULT_SUBSPACE_METHOD,
    n_components: int | str | None = DEFAULT_SUBSPACE_COMPONENTS,
    regularization: float | str = DEFAULT_SUBSPACE_REGULARIZATION,
    eigen_ridge: float | str = DEFAULT_SUBSPACE_EIGEN_RIDGE,
    standardize: bool = True,
    class_balance_source: bool = False,
    normalize_latent: bool = False,
) -> SubspaceAdaptationConfig:
    """Normalize public configuration values."""

    normalized_method = normalize_subspace_method(method)
    balance = bool(class_balance_source or normalized_method == "balanced_tca")
    return SubspaceAdaptationConfig(
        method=normalized_method,
        n_components=_normalize_components_request(n_components),
        regularization=_nonnegative_float(regularization, name="regularization"),
        eigen_ridge=_positive_float(eigen_ridge, name="eigen_ridge"),
        standardize=bool(standardize),
        class_balance_source=balance,
        normalize_latent=bool(normalize_latent),
    )


def normalize_subspace_method(method: str | None) -> str:
    """Normalize aliases for subspace-adaptation methods."""

    normalized = DEFAULT_SUBSPACE_METHOD if method is None else str(method).strip().lower().replace("-", "_")
    normalized = {
        "transfer_component_analysis": "tca",
        "marginal_tca": "tca",
        "source_balanced_tca": "balanced_tca",
        "class_balanced_tca": "balanced_tca",
        "balanced_transfer_component_analysis": "balanced_tca",
    }.get(normalized, normalized)
    if normalized not in SUBSPACE_METHODS:
        raise ValueError(f"Unknown subspace method {method!r}. Available methods: {', '.join(SUBSPACE_METHODS)}.")
    return normalized


def _resolve_config(config: SubspaceAdaptationConfig | dict[str, Any] | None, **overrides: Any) -> SubspaceAdaptationConfig:
    raw = {} if config is None else (dict(config) if isinstance(config, dict) else {
        "method": config.method,
        "n_components": config.n_components,
        "regularization": config.regularization,
        "eigen_ridge": config.eigen_ridge,
        "standardize": config.standardize,
        "class_balance_source": config.class_balance_source,
        "normalize_latent": config.normalize_latent,
    })
    for key, value in overrides.items():
        if value is not None:
            raw[key] = value
    return subspace_adaptation_config(**raw)


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    if matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one row and one feature column.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _object_vector(values: Sequence[Any] | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    try:
        items = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a one-dimensional sequence.") from exc
    if len(items) != expected_length:
        raise ValueError(f"{name} must contain one value per source row: {len(items)} != {expected_length}.")
    vector = np.empty(len(items), dtype=object)
    for index, value in enumerate(items):
        vector[index] = value
    return vector


def _source_weights(n_rows: int, *, labels: np.ndarray | None, class_balance: bool) -> np.ndarray:
    if not class_balance:
        return np.full(n_rows, 1.0 / float(n_rows), dtype=float)
    if labels is None:
        raise ValueError("class-balanced source weights require source labels.")
    classes = tuple(dict.fromkeys(labels.tolist()))
    weights = np.zeros(n_rows, dtype=float)
    for class_label in classes:
        mask = _object_mask(labels, class_label)
        weights[mask] = 1.0 / (float(len(classes)) * float(np.count_nonzero(mask)))
    return weights / float(np.sum(weights))


def _object_mask(values: np.ndarray, target: Any) -> np.ndarray:
    return np.asarray([_object_equal(value, target) for value in values.tolist()], dtype=bool)


def _object_equal(left: Any, right: Any) -> bool:
    result = left == right
    if isinstance(result, np.ndarray):
        return bool(np.array_equal(left, right))
    return bool(result)


def _weighted_mean_gap(source: np.ndarray, target: np.ndarray, *, source_weights: np.ndarray, target_weights: np.ndarray) -> float:
    source_mean = source_weights.reshape(1, -1) @ source
    target_mean = target_weights.reshape(1, -1) @ target
    return float(np.linalg.norm(source_mean.ravel() - target_mean.ravel()))


def _effective_components(value: int | str, *, n_samples: int, n_features: int) -> int:
    requested = _normalize_components_request(value)
    maximum = max(1, min(int(n_features), int(n_samples) - 1))
    if requested == "all":
        return maximum
    return min(int(requested), maximum)


def _normalize_components_request(value: int | str | None) -> int | str:
    if value is None:
        return DEFAULT_SUBSPACE_COMPONENTS
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"all", "full", "inf", "infinity"}:
            return "all"
        value = text
    parsed = float(value)
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 1:
        raise ValueError("n_components must be a positive integer or 'all'.")
    return int(parsed)


def _canonicalize_projection(projection: np.ndarray) -> np.ndarray:
    result = np.asarray(projection, dtype=float).copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0.0:
            result[:, column] *= -1.0
    return result


def _metadata(
    *,
    cfg: SubspaceAdaptationConfig,
    n_source_rows: int,
    n_target_rows: int,
    feature_dim: int,
    n_components: int,
    source_labels_used: bool,
    raw_gap: float,
    latent_gap: float,
    eigenvalues: np.ndarray,
) -> dict[str, Any]:
    return {
        "subspace_adaptation": True,
        "subspace_adaptation_protocol": SUBSPACE_ADAPTATION_PROTOCOL,
        "subspace_adaptation_protocol_category": SUBSPACE_ADAPTATION_CATEGORY,
        "subspace_adaptation_method": cfg.method,
        "subspace_adaptation_uses_source_features": True,
        "subspace_adaptation_uses_source_labels": bool(source_labels_used),
        "subspace_adaptation_uses_target_features": True,
        "subspace_adaptation_uses_target_labels": False,
        "subspace_adaptation_valid_for_strict_source_only": False,
        "subspace_adaptation_valid_for_unlabeled_target_adaptation": True,
        "subspace_adaptation_valid_for_target_calibration": False,
        "subspace_adaptation_n_source_rows": int(n_source_rows),
        "subspace_adaptation_n_target_rows": int(n_target_rows),
        "subspace_adaptation_feature_dim": int(feature_dim),
        "subspace_adaptation_n_components": int(n_components),
        "subspace_adaptation_regularization": float(cfg.regularization),
        "subspace_adaptation_eigen_ridge": float(cfg.eigen_ridge),
        "subspace_adaptation_standardize": bool(cfg.standardize),
        "subspace_adaptation_class_balance_source": bool(cfg.class_balance_source),
        "subspace_adaptation_normalize_latent": bool(cfg.normalize_latent),
        "subspace_adaptation_raw_mean_gap": float(raw_gap),
        "subspace_adaptation_latent_mean_gap": float(latent_gap),
        "subspace_adaptation_eigenvalues": "|".join(f"{float(v):.12g}" for v in eigenvalues),
    }


def _positive_float(value: float | str, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be positive and finite.")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed


def _nonnegative_float(value: float | str, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and non-negative.")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return parsed
