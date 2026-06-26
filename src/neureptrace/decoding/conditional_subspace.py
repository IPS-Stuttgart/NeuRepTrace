"""Joint distribution adaptation for unlabeled target rows."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh
from neureptrace.decoding.subspace_adaptation import MIN_SCALE, _canonicalize_projection, _effective_components, _feature_matrix, _object_mask, _object_vector

@dataclass(frozen=True, slots=True)
class JDAResult:
    source_features: np.ndarray
    target_features: np.ndarray
    pseudo_labels: np.ndarray
    projection: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    metadata: dict


def fit_jda(source_features, source_labels, target_features, *, n_components=16, regularization=1e-3, conditional_weight=1.0, max_iterations=10):
    """Fit Category-2 JDA using source labels and target pseudo labels."""
    source = _feature_matrix(source_features, name="source_features")
    target = _feature_matrix(target_features, name="target_features")
    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target feature widths differ")
    labels = _object_vector(source_labels, expected_length=source.shape[0], name="source_labels")
    classes = tuple(dict.fromkeys(labels.tolist()))
    if len(classes) < 2:
        raise ValueError("JDA requires at least two source classes")
    iterations = _positive_int(max_iterations, "max_iterations")
    reg, weight = _nonnegative(regularization, "regularization"), _nonnegative(conditional_weight, "conditional_weight")
    joint = np.vstack([source, target]).astype(float)
    mean, scale = joint.mean(0), np.maximum(joint.std(0, ddof=1), MIN_SCALE)
    z = (joint - mean) / scale
    ns, nt = len(source), len(target)
    pseudo = _predict(z[:ns], labels, z[ns:], classes)
    h = np.eye(ns + nt) - np.full((ns + nt, ns + nt), 1 / (ns + nt))
    k = _effective_components(n_components, n_samples=ns + nt, n_features=z.shape[1])
    projection, converged = np.eye(z.shape[1], k), False
    for iteration in range(1, iterations + 1):
        domain = np.r_[np.full(ns, 1 / ns), np.full(nt, -1 / nt)]
        discrepancy = np.outer(domain, domain) + weight * _conditional(labels, pseudo, classes, ns, nt)
        norm = np.linalg.norm(discrepancy)
        if norm > MIN_SCALE: discrepancy /= norm
        a = z.T @ discrepancy @ z + reg * np.eye(z.shape[1])
        b = z.T @ h @ z + 1e-6 * np.eye(z.shape[1])
        values, vectors = eigh(a, b)
        projection = _canonicalize_projection(vectors[:, np.argsort(values)[:k]])
        latent = z @ projection
        updated = _predict(latent[:ns], labels, latent[ns:], classes)
        if _same(updated, pseudo):
            pseudo, converged = updated, True
            break
        pseudo = updated
    latent = z @ projection
    metadata = {
        "jda_protocol_category": "2_unlabeled_target_adaptive",
        "jda_uses_source_labels": True,
        "jda_uses_target_features": True,
        "jda_uses_target_labels": False,
        "jda_iterations": iteration,
        "jda_converged": converged,
    }
    return JDAResult(latent[:ns].astype(np.float32), latent[ns:].astype(np.float32), pseudo,
                     projection.astype(np.float32), mean.astype(np.float32), scale.astype(np.float32), metadata)


def transform_jda(features, result):
    matrix = _feature_matrix(features, name="features")
    return (((matrix - result.mean) / result.scale) @ result.projection).astype(np.float32)


def _conditional(source_labels, pseudo, classes, ns, nt):
    matrix = np.zeros((ns + nt, ns + nt))
    for label in classes:
        sm, tm = _object_mask(source_labels, label), _object_mask(pseudo, label)
        sc, tc = sm.sum(), tm.sum()
        if sc and tc:
            vector = np.zeros(ns + nt)
            vector[:ns][sm], vector[ns:][tm] = 1 / sc, -1 / tc
            matrix += np.outer(vector, vector)
    return matrix


def _predict(source, labels, target, classes):
    centers = np.vstack([source[_object_mask(labels, label)].mean(0) for label in classes])
    distance = ((target[:, None, :] - centers[None, :, :]) ** 2).sum(2)
    result = np.empty(len(target), dtype=object)
    for row, index in enumerate(distance.argmin(1)):
        result[row] = classes[int(index)]
    return result


def _same(left, right):
    return left.shape == right.shape and all(a == b for a, b in zip(left, right, strict=True))


def _positive_int(value, name):
    value = float(value)
    if not np.isfinite(value) or value % 1 or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative(value, name):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value
