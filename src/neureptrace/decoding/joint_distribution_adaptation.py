"""Iterative Joint Distribution Adaptation for cross-subject feature transfer.

The implementation is intentionally protocol-explicit. It uses labeled source
features and unlabeled target features. Target class structure is represented by
pseudo-labels or optional source-model target probabilities; held-out target labels
are not accepted by the public API.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.linalg import eigh

JDA_PROTOCOL = "unlabeled_target_joint_distribution_adaptation"
JDA_CATEGORY = "2_unlabeled_target_adaptive"
JDA_METHODS = ("jda", "soft_jda")
_MIN_SCALE = 1e-12


@dataclass(frozen=True, slots=True)
class JointDistributionAdaptationConfig:
    """Configuration for iterative JDA."""

    method: str = "jda"
    n_components: int | str = 16
    max_iterations: int = 10
    conditional_weight: float = 1.0
    regularization: float = 1e-3
    eigen_ridge: float = 1e-6
    temperature: float = 1.0
    standardize: bool = True
    normalize_latent: bool = False


@dataclass(frozen=True, slots=True)
class JointDistributionAdaptationResult:
    """Projected source/target rows and pseudo-label diagnostics."""

    source_features: np.ndarray
    target_features: np.ndarray
    projection: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    eigenvalues: np.ndarray
    target_pseudo_labels: np.ndarray
    target_probabilities: np.ndarray
    classes: tuple[Any, ...]
    n_iterations: int
    converged: bool
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments,too-many-locals

def fit_joint_distribution_adaptation(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    target_features: Sequence[Sequence[float]] | np.ndarray,
    *,
    target_probabilities: Sequence[Sequence[float]] | np.ndarray | None = None,
    classes: Sequence[Any] | np.ndarray | None = None,
    config: JointDistributionAdaptationConfig | dict[str, Any] | None = None,
    method: str | None = None,
    n_components: int | str | None = None,
    max_iterations: int | str | None = None,
    conditional_weight: float | str | None = None,
    regularization: float | str | None = None,
    eigen_ridge: float | str | None = None,
    temperature: float | str | None = None,
    standardize: bool | None = None,
    normalize_latent: bool | None = None,
) -> JointDistributionAdaptationResult:
    """Fit iterative marginal-plus-conditional source-target alignment.

    ``target_probabilities`` may contain source-model probabilities for the
    unlabeled target rows. If omitted, target pseudo-labels are initialized by
    nearest source-class centroids. No target-label argument exists.
    """

    cfg = _resolve_config(
        config,
        method=method,
        n_components=n_components,
        max_iterations=max_iterations,
        conditional_weight=conditional_weight,
        regularization=regularization,
        eigen_ridge=eigen_ridge,
        temperature=temperature,
        standardize=standardize,
        normalize_latent=normalize_latent,
    )
    source = _feature_matrix(source_features, name="source_features")
    target = _feature_matrix(target_features, name="target_features")
    if source.shape[1] != target.shape[1]:
        raise ValueError("source_features and target_features must have the same feature width.")
    labels = _object_vector(source_labels, expected_length=source.shape[0], name="source_labels")
    class_order = _resolve_classes(labels, classes)
    if len(class_order) < 2:
        raise ValueError("Joint distribution adaptation requires at least two source classes.")
    encoded_source = _encode_labels(labels, class_order)

    joint = np.vstack([source, target]).astype(float, copy=False)
    mean = np.mean(joint, axis=0) if cfg.standardize else np.zeros(joint.shape[1], dtype=float)
    centered = joint - mean
    scale = np.std(centered, axis=0, ddof=1 if joint.shape[0] > 1 else 0) if cfg.standardize else np.ones(joint.shape[1], dtype=float)
    scale = np.maximum(scale, _MIN_SCALE)
    z = centered / scale
    source_z = z[: source.shape[0]]
    target_z = z[source.shape[0] :]

    if target_probabilities is None:
        responsibilities = _centroid_probabilities(source_z, encoded_source, target_z, len(class_order), temperature=cfg.temperature)
        used_initial_probabilities = False
    else:
        responsibilities = _probability_matrix(target_probabilities, expected_rows=target.shape[0], expected_classes=len(class_order))
        used_initial_probabilities = True
    pseudo = np.argmax(responsibilities, axis=1)

    centering = np.eye(joint.shape[0], dtype=float) - np.full((joint.shape[0], joint.shape[0]), 1.0 / float(joint.shape[0]))
    feature_dim = z.shape[1]
    component_count = _effective_components(cfg.n_components, n_samples=z.shape[0], n_features=feature_dim)
    converged = False
    projection = np.eye(feature_dim, component_count, dtype=float)
    selected_values = np.zeros(component_count, dtype=float)
    source_latent = source_z @ projection
    target_latent = target_z @ projection

    for iteration in range(1, cfg.max_iterations + 1):
        discrepancy = _marginal_matrix(source.shape[0], target.shape[0])
        discrepancy += cfg.conditional_weight * _conditional_matrix(encoded_source, responsibilities, len(class_order))
        norm = float(np.linalg.norm(discrepancy, ord="fro"))
        if norm > _MIN_SCALE:
            discrepancy /= norm
        a_matrix = z.T @ discrepancy @ z + cfg.regularization * np.eye(feature_dim, dtype=float)
        b_matrix = z.T @ centering @ z + cfg.eigen_ridge * np.eye(feature_dim, dtype=float)
        values, vectors = eigh(a_matrix, b_matrix, check_finite=True)
        selected = np.argsort(values)[:component_count]
        selected_values = values[selected]
        projection = _canonicalize_projection(vectors[:, selected])
        latent = z @ projection
        if cfg.normalize_latent:
            latent_scale = np.maximum(np.std(latent, axis=0, ddof=1 if latent.shape[0] > 1 else 0), _MIN_SCALE)
            latent /= latent_scale
            projection /= latent_scale.reshape(1, -1)
        source_latent = latent[: source.shape[0]]
        target_latent = latent[source.shape[0] :]
        updated_probabilities = _centroid_probabilities(
            source_latent,
            encoded_source,
            target_latent,
            len(class_order),
            temperature=cfg.temperature,
        )
        updated_pseudo = np.argmax(updated_probabilities, axis=1)
        converged = bool(np.array_equal(updated_pseudo, pseudo))
        pseudo = updated_pseudo
        responsibilities = updated_probabilities if cfg.method == "soft_jda" else _one_hot(pseudo, len(class_order))
        if converged:
            iterations_run = iteration
            break
    else:
        iterations_run = cfg.max_iterations

    pseudo_labels = _decode_labels(pseudo, class_order)
    metadata = _metadata(
        cfg=cfg,
        n_source_rows=source.shape[0],
        n_target_rows=target.shape[0],
        feature_dim=feature_dim,
        n_components=component_count,
        n_classes=len(class_order),
        iterations=iterations_run,
        converged=converged,
        used_initial_probabilities=used_initial_probabilities,
        pseudo=pseudo,
        eigenvalues=selected_values,
    )
    return JointDistributionAdaptationResult(
        source_features=source_latent.astype(np.float32, copy=False),
        target_features=target_latent.astype(np.float32, copy=False),
        projection=projection.astype(np.float32, copy=False),
        feature_mean=mean.astype(np.float32, copy=False),
        feature_scale=scale.astype(np.float32, copy=False),
        eigenvalues=np.asarray(selected_values, dtype=float),
        target_pseudo_labels=pseudo_labels,
        target_probabilities=np.asarray(responsibilities, dtype=np.float32),
        classes=class_order,
        n_iterations=int(iterations_run),
        converged=converged,
        metadata=metadata,
    )


def transform_joint_distribution_features(features: Sequence[Sequence[float]] | np.ndarray, result: JointDistributionAdaptationResult) -> np.ndarray:
    """Transform new rows with a fitted JDA projection."""

    matrix = _feature_matrix(features, name="features")
    if matrix.shape[1] != result.projection.shape[0]:
        raise ValueError("features width does not match the fitted projection.")
    return (((matrix - result.feature_mean) / result.feature_scale) @ result.projection).astype(np.float32, copy=False)


def joint_distribution_adaptation_config(
    *,
    method: str | None = "jda",
    n_components: int | str | None = 16,
    max_iterations: int | str = 10,
    conditional_weight: float | str = 1.0,
    regularization: float | str = 1e-3,
    eigen_ridge: float | str = 1e-6,
    temperature: float | str = 1.0,
    standardize: bool = True,
    normalize_latent: bool = False,
) -> JointDistributionAdaptationConfig:
    """Normalize JDA configuration values."""

    return JointDistributionAdaptationConfig(
        method=normalize_jda_method(method),
        n_components=_normalize_components(n_components),
        max_iterations=_positive_int(max_iterations, name="max_iterations"),
        conditional_weight=_nonnegative_float(conditional_weight, name="conditional_weight"),
        regularization=_nonnegative_float(regularization, name="regularization"),
        eigen_ridge=_positive_float(eigen_ridge, name="eigen_ridge"),
        temperature=_positive_float(temperature, name="temperature"),
        standardize=bool(standardize),
        normalize_latent=bool(normalize_latent),
    )


def normalize_jda_method(method: str | None) -> str:
    """Normalize public JDA method aliases."""

    normalized = "jda" if method is None else str(method).strip().lower().replace("-", "_")
    normalized = {
        "joint_distribution_adaptation": "jda",
        "hard_jda": "jda",
        "soft": "soft_jda",
        "probabilistic_jda": "soft_jda",
        "soft_joint_distribution_adaptation": "soft_jda",
    }.get(normalized, normalized)
    if normalized not in JDA_METHODS:
        raise ValueError(f"Unknown JDA method {method!r}. Available methods: {', '.join(JDA_METHODS)}.")
    return normalized


def _resolve_config(config: JointDistributionAdaptationConfig | dict[str, Any] | None, **overrides: Any) -> JointDistributionAdaptationConfig:
    if config is None:
        raw: dict[str, Any] = {}
    elif isinstance(config, dict):
        raw = dict(config)
    else:
        raw = {name: getattr(config, name) for name in config.__dataclass_fields__}
    raw.update({key: value for key, value in overrides.items() if value is not None})
    return joint_distribution_adaptation_config(**raw)


def _marginal_matrix(n_source: int, n_target: int) -> np.ndarray:
    vector = np.concatenate([
        np.full(n_source, 1.0 / n_source, dtype=float),
        np.full(n_target, -1.0 / n_target, dtype=float),
    ])
    return np.outer(vector, vector)


def _conditional_matrix(source_labels: np.ndarray, target_responsibilities: np.ndarray, n_classes: int) -> np.ndarray:
    n_source = source_labels.shape[0]
    n_target = target_responsibilities.shape[0]
    matrix = np.zeros((n_source + n_target, n_source + n_target), dtype=float)
    for class_index in range(n_classes):
        source_mask = source_labels == class_index
        target_weights = np.asarray(target_responsibilities[:, class_index], dtype=float)
        if not np.any(source_mask) or float(np.sum(target_weights)) <= _MIN_SCALE:
            continue
        vector = np.zeros(n_source + n_target, dtype=float)
        vector[:n_source][source_mask] = 1.0 / float(np.count_nonzero(source_mask))
        vector[n_source:] = -target_weights / float(np.sum(target_weights))
        matrix += np.outer(vector, vector)
    return matrix


def _centroid_probabilities(source: np.ndarray, source_labels: np.ndarray, target: np.ndarray, n_classes: int, *, temperature: float) -> np.ndarray:
    centroids = np.vstack([np.mean(source[source_labels == index], axis=0) for index in range(n_classes)])
    distances = np.sum((target[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    logits = -distances / max(float(temperature), _MIN_SCALE)
    logits -= np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(np.clip(logits, -50.0, 50.0))
    return probabilities / np.sum(probabilities, axis=1, keepdims=True)


def _one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    result = np.zeros((labels.shape[0], n_classes), dtype=float)
    result[np.arange(labels.shape[0]), labels] = 1.0
    return result


def _resolve_classes(labels: np.ndarray, classes: Sequence[Any] | np.ndarray | None) -> tuple[Any, ...]:
    inferred = tuple(dict.fromkeys(labels.tolist()))
    if classes is None:
        return inferred
    supplied = tuple(_object_vector(classes, expected_length=None, name="classes").tolist())
    if len(set(supplied)) != len(supplied):
        raise ValueError("classes must be unique.")
    missing = [value for value in inferred if value not in supplied]
    if missing:
        raise ValueError(f"classes omit source label(s): {missing!r}.")
    return supplied


def _encode_labels(labels: np.ndarray, classes: tuple[Any, ...]) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(classes)}
    return np.asarray([lookup[value] for value in labels.tolist()], dtype=int)


def _decode_labels(labels: np.ndarray, classes: tuple[Any, ...]) -> np.ndarray:
    decoded = np.empty(labels.shape[0], dtype=object)
    for index, value in enumerate(labels.tolist()):
        decoded[index] = classes[value]
    return decoded


def _probability_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, expected_rows: int, expected_classes: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (expected_rows, expected_classes):
        raise ValueError(f"target_probabilities must have shape ({expected_rows}, {expected_classes}).")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("target_probabilities must contain finite non-negative values.")
    row_sums = np.sum(matrix, axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("target probability rows must have positive mass.")
    return matrix / row_sums


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values.")
    return matrix


def _object_vector(values: Sequence[Any] | np.ndarray, *, expected_length: int | None, name: str) -> np.ndarray:
    items = list(values)
    if expected_length is not None and len(items) != expected_length:
        raise ValueError(f"{name} must contain {expected_length} values, got {len(items)}.")
    vector = np.empty(len(items), dtype=object)
    for index, value in enumerate(items):
        hash(value)
        vector[index] = value
    return vector


def _effective_components(value: int | str, *, n_samples: int, n_features: int) -> int:
    requested = _normalize_components(value)
    maximum = max(1, min(n_features, n_samples - 1))
    return maximum if requested == "all" else min(int(requested), maximum)


def _normalize_components(value: int | str | None) -> int | str:
    if value is None:
        return 16
    if isinstance(value, str) and value.strip().lower() in {"all", "full", "inf", "infinity"}:
        return "all"
    return _positive_int(value, name="n_components")


def _canonicalize_projection(projection: np.ndarray) -> np.ndarray:
    result = np.asarray(projection, dtype=float).copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0.0:
            result[:, column] *= -1.0
    return result


def _metadata(*, cfg: JointDistributionAdaptationConfig, n_source_rows: int, n_target_rows: int, feature_dim: int, n_components: int, n_classes: int, iterations: int, converged: bool, used_initial_probabilities: bool, pseudo: np.ndarray, eigenvalues: np.ndarray) -> dict[str, Any]:
    counts = np.bincount(pseudo, minlength=n_classes)
    return {
        "joint_distribution_adaptation": True,
        "jda_protocol": JDA_PROTOCOL,
        "jda_protocol_category": JDA_CATEGORY,
        "jda_method": cfg.method,
        "jda_uses_source_features": True,
        "jda_uses_source_labels": True,
        "jda_uses_target_features": True,
        "jda_uses_target_probabilities": bool(used_initial_probabilities),
        "jda_uses_target_labels": False,
        "jda_valid_for_strict_source_only": False,
        "jda_valid_for_unlabeled_target_adaptation": True,
        "jda_n_source_rows": int(n_source_rows),
        "jda_n_target_rows": int(n_target_rows),
        "jda_feature_dim": int(feature_dim),
        "jda_n_components": int(n_components),
        "jda_n_classes": int(n_classes),
        "jda_n_iterations": int(iterations),
        "jda_converged": bool(converged),
        "jda_conditional_weight": float(cfg.conditional_weight),
        "jda_regularization": float(cfg.regularization),
        "jda_eigen_ridge": float(cfg.eigen_ridge),
        "jda_temperature": float(cfg.temperature),
        "jda_pseudo_label_counts": "|".join(str(int(value)) for value in counts),
        "jda_eigenvalues": "|".join(f"{float(value):.12g}" for value in eigenvalues),
    }


def _positive_int(value: int | str, *, name: str) -> int:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(parsed)


def _positive_float(value: float | str, *, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed


def _nonnegative_float(value: float | str, *, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return parsed
