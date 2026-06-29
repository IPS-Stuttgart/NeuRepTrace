"""Unlabeled target probability smoothing.

This module smooths already predicted probability rows over a graph built from
unlabeled held-out feature rows.  It is a Category-2 post-processing helper: the
feature geometry of the held-out batch may affect the probabilities, but held-out
labels are not accepted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

TARGET_PROBABILITY_SMOOTHING_PROTOCOL = "unlabeled_target_probability_smoothing"
TARGET_PROBABILITY_SMOOTHING_CATEGORY = "2_unlabeled_target_adaptive"
DEFAULT_ALPHA = 0.5
DEFAULT_EPSILON = 1e-12
DEFAULT_MAX_ITER = 50
DEFAULT_TOL = 1e-7


@dataclass(frozen=True, slots=True)
class TargetProbabilitySmoothingConfig:
    """Configuration for graph smoothing of probability rows."""

    alpha: float = DEFAULT_ALPHA
    gamma: float | str = "auto"
    n_neighbors: int | None = None
    max_iter: int = DEFAULT_MAX_ITER
    tol: float = DEFAULT_TOL
    standardize: bool = True
    epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True, slots=True)
class TargetProbabilitySmoothingResult:
    """Smoothed probabilities and provenance."""

    probabilities: np.ndarray
    initial_probabilities: np.ndarray
    affinity: np.ndarray
    n_iter: int
    converged: bool
    metadata: dict[str, Any] = field(default_factory=dict)


def smooth_target_probabilities(
    target_features: Sequence[Sequence[float]] | np.ndarray,
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    config: TargetProbabilitySmoothingConfig | Mapping[str, Any] | None = None,
) -> TargetProbabilitySmoothingResult:
    """Smooth probability rows over an unlabeled target-feature graph.

    Parameters
    ----------
    target_features:
        Held-out feature rows used only to build the smoothing graph.
    probabilities:
        Initial probability rows from a source-trained decoder.
    config:
        Graph and smoothing options.
    """

    cfg = target_probability_smoothing_config() if config is None else _coerce_config(config)
    features = _feature_matrix(target_features, name="target_features")
    initial = _probability_matrix(probabilities, expected_rows=features.shape[0], epsilon=cfg.epsilon)
    prepared = _standardize(features, enabled=cfg.standardize, epsilon=cfg.epsilon)
    affinity, gamma = rbf_affinity(prepared, gamma=cfg.gamma, n_neighbors=cfg.n_neighbors, epsilon=cfg.epsilon)
    transition = row_normalize(affinity, epsilon=cfg.epsilon)
    smoothed, n_iter, converged = _iterate(transition, initial, alpha=cfg.alpha, max_iter=cfg.max_iter, tol=cfg.tol, epsilon=cfg.epsilon)
    metadata = {
        "target_probability_smoothing": True,
        "target_probability_smoothing_protocol": TARGET_PROBABILITY_SMOOTHING_PROTOCOL,
        "target_probability_smoothing_protocol_category": TARGET_PROBABILITY_SMOOTHING_CATEGORY,
        "target_probability_smoothing_uses_target_features": True,
        "target_probability_smoothing_uses_target_labels": False,
        "target_probability_smoothing_valid_for_strict_source_only": False,
        "target_probability_smoothing_valid_for_unlabeled_target_adaptation": True,
        "target_probability_smoothing_valid_for_benchmark": False,
        "target_probability_smoothing_n_target_rows": int(features.shape[0]),
        "target_probability_smoothing_feature_dim": int(features.shape[1]),
        "target_probability_smoothing_n_classes": int(initial.shape[1]),
        "target_probability_smoothing_alpha": float(cfg.alpha),
        "target_probability_smoothing_gamma": float(gamma),
        "target_probability_smoothing_n_neighbors": "" if cfg.n_neighbors is None else int(cfg.n_neighbors),
        "target_probability_smoothing_max_iter": int(cfg.max_iter),
        "target_probability_smoothing_n_iter": int(n_iter),
        "target_probability_smoothing_converged": bool(converged),
        "target_probability_smoothing_standardize": bool(cfg.standardize),
    }
    return TargetProbabilitySmoothingResult(
        probabilities=smoothed.astype(np.float32, copy=False),
        initial_probabilities=initial.astype(np.float32, copy=False),
        affinity=affinity.astype(np.float32, copy=False),
        n_iter=int(n_iter),
        converged=bool(converged),
        metadata=metadata,
    )


def target_probability_smoothing_config(
    *,
    alpha: float | str = DEFAULT_ALPHA,
    gamma: float | str = "auto",
    n_neighbors: int | str | None = None,
    max_iter: int | str = DEFAULT_MAX_ITER,
    tol: float | str = DEFAULT_TOL,
    standardize: bool | int | str = True,
    epsilon: float | str = DEFAULT_EPSILON,
) -> TargetProbabilitySmoothingConfig:
    """Normalize smoothing options."""

    return TargetProbabilitySmoothingConfig(
        alpha=_closed_unit_float(alpha, name="alpha"),
        gamma=_normalize_gamma(gamma),
        n_neighbors=None if n_neighbors in {None, "", "none", "None"} else _positive_int(n_neighbors, name="n_neighbors"),
        max_iter=_positive_int(max_iter, name="max_iter"),
        tol=_positive_float(tol, name="tol"),
        standardize=_bool_value(standardize, name="standardize"),
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


def rbf_affinity(features: Sequence[Sequence[float]] | np.ndarray, *, gamma: float | str = "auto", n_neighbors: int | None = None, epsilon: float = DEFAULT_EPSILON) -> tuple[np.ndarray, float]:
    """Build a symmetric RBF affinity graph."""

    matrix = _feature_matrix(features, name="features")
    squared = _squared_euclidean(matrix, matrix)
    resolved_gamma = _auto_gamma(squared, epsilon=epsilon) if gamma == "auto" else _positive_float(gamma, name="gamma")
    affinity = np.exp(-resolved_gamma * squared)
    np.fill_diagonal(affinity, 0.0)
    if n_neighbors is not None and affinity.shape[0] > 1:
        k = min(_positive_int(n_neighbors, name="n_neighbors"), affinity.shape[0] - 1)
        keep = np.zeros_like(affinity, dtype=bool)
        for row in range(affinity.shape[0]):
            keep[row, np.argsort(affinity[row])[-k:]] = True
        affinity = np.where(keep | keep.T, affinity, 0.0)
    return affinity, float(resolved_gamma)


def row_normalize(matrix: Sequence[Sequence[float]] | np.ndarray, *, epsilon: float = DEFAULT_EPSILON) -> np.ndarray:
    """Return a row-stochastic matrix with self-loops for isolated rows."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square.")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("matrix must contain finite non-negative values.")
    row_sums = values.sum(axis=1, keepdims=True)
    output = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums > float(epsilon))
    isolated = np.ravel(row_sums <= float(epsilon))
    if np.any(isolated):
        output[isolated, isolated] = 1.0
    return output


def _iterate(transition: np.ndarray, initial: np.ndarray, *, alpha: float, max_iter: int, tol: float, epsilon: float) -> tuple[np.ndarray, int, bool]:
    current = initial.copy()
    for iteration in range(1, max_iter + 1):
        updated = alpha * (transition @ current) + (1.0 - alpha) * initial
        updated = _normalize_probability_rows(updated, epsilon=epsilon)
        if float(np.max(np.abs(updated - current))) <= tol:
            return updated, iteration, True
        current = updated
    return current, max_iter, False


def _coerce_config(config: TargetProbabilitySmoothingConfig | Mapping[str, Any]) -> TargetProbabilitySmoothingConfig:
    if isinstance(config, TargetProbabilitySmoothingConfig):
        return config
    return target_probability_smoothing_config(**dict(config))


def _standardize(features: np.ndarray, *, enabled: bool, epsilon: float) -> np.ndarray:
    if not enabled:
        return features.astype(float, copy=False)
    mean = np.mean(features, axis=0)
    scale = np.maximum(np.std(features - mean, axis=0, ddof=1 if features.shape[0] > 1 else 0), float(epsilon))
    return (features - mean) / scale


def _probability_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, expected_rows: int, epsilon: float) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != expected_rows:
        raise ValueError(f"probabilities must have {expected_rows} rows.")
    return _normalize_probability_rows(matrix, epsilon=epsilon)


def _normalize_probability_rows(values: np.ndarray, *, epsilon: float) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("probabilities must be finite and non-negative.")
    matrix = np.maximum(matrix, float(epsilon))
    return matrix / matrix.sum(axis=1, keepdims=True)


def _auto_gamma(squared_distances: np.ndarray, *, epsilon: float) -> float:
    positive = squared_distances[squared_distances > float(epsilon)]
    return 1.0 if positive.size == 0 else float(1.0 / (2.0 * max(float(np.median(positive)), float(epsilon))))


def _squared_euclidean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.maximum(np.sum(left * left, axis=1, keepdims=True) + np.sum(right * right, axis=1, keepdims=True).T - 2.0 * (left @ right.T), 0.0)


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values.")
    return matrix


def _normalize_gamma(value: float | str) -> float | str:
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "auto":
            return "auto"
        value = text
    return _positive_float(value, name="gamma")


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


def _closed_unit_float(value: float | str, *, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
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
