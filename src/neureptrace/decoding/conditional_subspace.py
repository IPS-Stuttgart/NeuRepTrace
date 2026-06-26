"""Joint distribution adaptation for unlabeled target rows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import eigh

from neureptrace.decoding.subspace_adaptation import (
    MIN_SCALE,
    _canonicalize_projection,
    _effective_components,
    _feature_matrix,
    _object_mask,
)


@dataclass(frozen=True, slots=True)
class JDAResult:
    source_features: np.ndarray
    target_features: np.ndarray
    pseudo_labels: np.ndarray
    projection: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    metadata: dict[str, Any]


def fit_jda(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    target_features: Sequence[Sequence[float]] | np.ndarray,
    *,
    n_components: int | str = 16,
    regularization: float | str = 1e-3,
    conditional_weight: float | str = 1.0,
    max_iterations: int | str = 10,
) -> JDAResult:
    """Fit Category-2 JDA using source labels and target pseudo-labels."""

    source = _feature_matrix(source_features, name="source_features")
    target = _feature_matrix(target_features, name="target_features")
    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target feature widths differ")
    labels = _label_vector(source_labels, expected_length=source.shape[0], name="source_labels")
    classes = tuple(dict.fromkeys(labels.tolist()))
    if len(classes) < 2:
        raise ValueError("JDA requires at least two source classes")
    iterations = _positive_int(max_iterations, "max_iterations")
    reg = _nonnegative(regularization, "regularization")
    weight = _nonnegative(conditional_weight, "conditional_weight")

    joint = np.vstack([source, target]).astype(float, copy=False)
    mean = joint.mean(axis=0)
    scale = np.maximum(joint.std(axis=0, ddof=1), MIN_SCALE)
    z = (joint - mean) / scale
    n_source, n_target = len(source), len(target)
    pseudo = _predict(z[:n_source], labels, z[n_source:], classes)
    centering = np.eye(n_source + n_target) - np.full((n_source + n_target, n_source + n_target), 1 / (n_source + n_target))
    component_count = _effective_components(n_components, n_samples=n_source + n_target, n_features=z.shape[1])
    projection = np.eye(z.shape[1], component_count)
    converged = False
    iterations_run = iterations

    for iteration in range(1, iterations + 1):
        domain = np.r_[np.full(n_source, 1 / n_source), np.full(n_target, -1 / n_target)]
        discrepancy = np.outer(domain, domain) + weight * _conditional(labels, pseudo, classes, n_source, n_target)
        norm = np.linalg.norm(discrepancy)
        if norm > MIN_SCALE:
            discrepancy /= norm
        a_matrix = z.T @ discrepancy @ z + reg * np.eye(z.shape[1])
        b_matrix = z.T @ centering @ z + 1e-6 * np.eye(z.shape[1])
        values, vectors = eigh(a_matrix, b_matrix, check_finite=True)
        projection = _canonicalize_projection(vectors[:, np.argsort(values)[:component_count]])
        latent = z @ projection
        updated = _predict(latent[:n_source], labels, latent[n_source:], classes)
        if _same(updated, pseudo):
            pseudo = updated
            converged = True
            iterations_run = iteration
            break
        pseudo = updated

    latent = z @ projection
    metadata = {
        "jda_protocol_category": "2_unlabeled_target_adaptive",
        "jda_uses_source_labels": True,
        "jda_uses_target_features": True,
        "jda_uses_target_labels": False,
        "jda_iterations": int(iterations_run),
        "jda_converged": bool(converged),
    }
    return JDAResult(
        latent[:n_source].astype(np.float32),
        latent[n_source:].astype(np.float32),
        pseudo,
        projection.astype(np.float32),
        mean.astype(np.float32),
        scale.astype(np.float32),
        metadata,
    )


def transform_jda(features: Sequence[Sequence[float]] | np.ndarray, result: JDAResult) -> np.ndarray:
    matrix = _feature_matrix(features, name="features")
    if matrix.shape[1] != result.projection.shape[0]:
        raise ValueError(f"features width {matrix.shape[1]} does not match fitted projection width {result.projection.shape[0]}.")
    return (((matrix - result.mean) / result.scale) @ result.projection).astype(np.float32)


def _conditional(source_labels: np.ndarray, pseudo: np.ndarray, classes: tuple[Any, ...], n_source: int, n_target: int) -> np.ndarray:
    matrix = np.zeros((n_source + n_target, n_source + n_target), dtype=float)
    for label in classes:
        source_mask = _object_mask(source_labels, label)
        target_mask = _object_mask(pseudo, label)
        source_count = int(np.count_nonzero(source_mask))
        target_count = int(np.count_nonzero(target_mask))
        if source_count and target_count:
            vector = np.zeros(n_source + n_target, dtype=float)
            vector[:n_source][source_mask] = 1.0 / source_count
            vector[n_source:][target_mask] = -1.0 / target_count
            matrix += np.outer(vector, vector)
    return matrix


def _predict(source: np.ndarray, labels: np.ndarray, target: np.ndarray, classes: tuple[Any, ...]) -> np.ndarray:
    centers = np.vstack([source[_object_mask(labels, label)].mean(axis=0) for label in classes])
    distance = ((target[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    result = np.empty(len(target), dtype=object)
    for row, index in enumerate(distance.argmin(axis=1)):
        result[row] = classes[int(index)]
    return result


def _label_vector(values: Sequence[Any] | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    try:
        items = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a one-dimensional sequence") from exc
    if len(items) != expected_length:
        raise ValueError(f"{name} must contain {expected_length} values, got {len(items)}")
    vector = np.empty(len(items), dtype=object)
    for index, value in enumerate(items):
        vector[index] = _hashable_label(value, name=name)
    return vector


def _hashable_label(value: Any, *, name: str) -> Any:
    try:
        hash(value)
    except TypeError:
        return _hashable_composite_label(value, name=name)
    return value


def _hashable_composite_label(value: Any, *, name: str) -> tuple[Any, ...]:
    try:
        items = list(value)
    except TypeError as exc:
        raise ValueError(f"{name} values must be hashable") from exc
    label = tuple(_hashable_label(item, name=name) for item in items)
    try:
        hash(label)
    except TypeError as exc:
        raise ValueError(f"{name} values must be hashable") from exc
    return label


def _same(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and all(_object_equal(left_item, right_item) for left_item, right_item in zip(left, right, strict=True))


def _object_equal(left: Any, right: Any) -> bool:
    result = left == right
    if isinstance(result, np.ndarray):
        return bool(np.array_equal(left, right))
    return bool(result)


def _positive_int(value: int | str, name: str) -> int:
    value = float(value)
    if not np.isfinite(value) or value % 1 or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative(value: float | str, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value
