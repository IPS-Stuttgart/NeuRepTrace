"""Unlabeled target-prior probability adjustment.

This module adjusts target probability rows by estimating a target class prior
from the probability rows themselves.  It is a Category-2 post-processing helper:
source-model probabilities on held-out target rows may be used, but held-out
target labels are not part of the API.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

TARGET_PRIOR_ADJUSTMENT_PROTOCOL = "unlabeled_target_prior_probability_adjustment"
TARGET_PRIOR_ADJUSTMENT_CATEGORY = "2_unlabeled_target_adaptive"
PRIOR_ESTIMATORS = ("mean", "em")
DEFAULT_MAX_ITER = 100
DEFAULT_TOL = 1e-8
DEFAULT_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class TargetPriorAdjustmentConfig:
    """Configuration for unlabeled target-prior adjustment."""

    estimator: str = "em"
    source_prior: str | Sequence[float] = "uniform"
    strength: float = 1.0
    max_iter: int = DEFAULT_MAX_ITER
    tol: float = DEFAULT_TOL
    epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True, slots=True)
class TargetPriorAdjustmentResult:
    """Adjusted probability rows and protocol metadata."""

    probabilities: np.ndarray
    original_probabilities: np.ndarray
    target_prior: np.ndarray
    source_prior: np.ndarray
    prior_ratio: np.ndarray
    n_iter: int
    converged: bool
    metadata: dict[str, Any] = field(default_factory=dict)


def adjust_target_probabilities_to_prior(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    config: TargetPriorAdjustmentConfig | Mapping[str, Any] | None = None,
) -> TargetPriorAdjustmentResult:
    """Adjust target probability rows using an unlabeled target-prior estimate.

    Parameters
    ----------
    probabilities:
        Target probability rows produced by a source-trained model.  Rows are
        normalized internally.
    config:
        Prior-adjustment options.  A mapping is normalized through
        :func:`target_prior_adjustment_config`.
    """

    cfg = target_prior_adjustment_config() if config is None else _coerce_config(config)
    original = _probability_matrix(probabilities, epsilon=cfg.epsilon)
    source_prior = _source_prior(cfg.source_prior, n_classes=original.shape[1], epsilon=cfg.epsilon)
    if cfg.estimator == "mean":
        target_prior = estimate_target_prior_mean(original, epsilon=cfg.epsilon)
        n_iter = 1
        converged = True
    elif cfg.estimator == "em":
        target_prior, n_iter, converged = estimate_target_prior_em(original, source_prior=source_prior, max_iter=cfg.max_iter, tol=cfg.tol, epsilon=cfg.epsilon)
    else:  # pragma: no cover - guarded by config normalization
        raise ValueError(f"Unhandled target-prior estimator {cfg.estimator!r}.")
    blended_prior = (1.0 - cfg.strength) * source_prior + cfg.strength * target_prior
    ratio = blended_prior / np.maximum(source_prior, cfg.epsilon)
    adjusted = _normalize_probability_rows(original * ratio[None, :], epsilon=cfg.epsilon)
    metadata = _metadata(cfg, n_rows=original.shape[0], n_classes=original.shape[1], n_iter=n_iter, converged=converged)
    return TargetPriorAdjustmentResult(
        probabilities=adjusted.astype(np.float32, copy=False),
        original_probabilities=original.astype(np.float32, copy=False),
        target_prior=blended_prior.astype(np.float32, copy=False),
        source_prior=source_prior.astype(np.float32, copy=False),
        prior_ratio=ratio.astype(np.float32, copy=False),
        n_iter=int(n_iter),
        converged=bool(converged),
        metadata=metadata,
    )


def estimate_target_prior_mean(probabilities: Sequence[Sequence[float]] | np.ndarray, *, epsilon: float = DEFAULT_EPSILON) -> np.ndarray:
    """Estimate a target prior by averaging probability rows."""

    matrix = _probability_matrix(probabilities, epsilon=epsilon)
    return _normalize_probability_rows(np.mean(matrix, axis=0, keepdims=True), epsilon=epsilon)[0]


def estimate_target_prior_em(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    source_prior: Sequence[float] | np.ndarray | None = None,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[np.ndarray, int, bool]:
    """Estimate an unlabeled target prior by EM-style prior reweighting."""

    matrix = _probability_matrix(probabilities, epsilon=epsilon)
    src = np.full(matrix.shape[1], 1.0 / matrix.shape[1], dtype=float) if source_prior is None else _prior_vector(source_prior, n_classes=matrix.shape[1], epsilon=epsilon)
    prior = estimate_target_prior_mean(matrix, epsilon=epsilon)
    iterations = _positive_int(max_iter, name="max_iter")
    tolerance = _positive_float(tol, name="tol")
    for iteration in range(1, iterations + 1):
        ratio = prior / np.maximum(src, float(epsilon))
        posterior = _normalize_probability_rows(matrix * ratio[None, :], epsilon=epsilon)
        new_prior = _normalize_probability_rows(np.mean(posterior, axis=0, keepdims=True), epsilon=epsilon)[0]
        delta = float(np.max(np.abs(new_prior - prior)))
        prior = new_prior
        if delta <= tolerance:
            return prior, iteration, True
    return prior, iterations, False


def target_prior_adjustment_config(
    *,
    estimator: str | None = "em",
    source_prior: str | Sequence[float] = "uniform",
    strength: float | str = 1.0,
    max_iter: int | str = DEFAULT_MAX_ITER,
    tol: float | str = DEFAULT_TOL,
    epsilon: float | str = DEFAULT_EPSILON,
) -> TargetPriorAdjustmentConfig:
    """Normalize public target-prior adjustment options."""

    return TargetPriorAdjustmentConfig(
        estimator=normalize_prior_estimator(estimator),
        source_prior=source_prior,
        strength=_unit_interval_float(strength, name="strength"),
        max_iter=_positive_int(max_iter, name="max_iter"),
        tol=_positive_float(tol, name="tol"),
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


def normalize_prior_estimator(value: str | None) -> str:
    """Normalize target-prior estimator aliases."""

    normalized = "em" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {"average": "mean", "avg": "mean", "expectation_maximization": "em"}.get(normalized, normalized)
    if normalized not in PRIOR_ESTIMATORS:
        raise ValueError(f"Unknown target prior estimator {value!r}.")
    return normalized


def _coerce_config(config: TargetPriorAdjustmentConfig | Mapping[str, Any]) -> TargetPriorAdjustmentConfig:
    if isinstance(config, TargetPriorAdjustmentConfig):
        return config
    return target_prior_adjustment_config(**dict(config))


def _source_prior(value: str | Sequence[float], *, n_classes: int, epsilon: float) -> np.ndarray:
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_")
        if normalized in {"uniform", "balanced", "flat"}:
            return np.full(n_classes, 1.0 / n_classes, dtype=float)
        raise ValueError("source_prior must be 'uniform' or an explicit prior vector.")
    return _prior_vector(value, n_classes=n_classes, epsilon=epsilon)


def _prior_vector(value: Sequence[float] | np.ndarray, *, n_classes: int, epsilon: float) -> np.ndarray:
    prior = np.asarray(value, dtype=float).reshape(-1)
    if prior.shape[0] != n_classes:
        raise ValueError(f"source_prior must contain one value per class: {prior.shape[0]} != {n_classes}.")
    if not np.all(np.isfinite(prior)) or np.any(prior < 0.0) or float(np.sum(prior)) <= 0.0:
        raise ValueError("source_prior must contain finite non-negative values with positive mass.")
    return _normalize_probability_rows(prior[None, :], epsilon=epsilon)[0]


def _probability_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, epsilon: float) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 2:
        raise ValueError("probabilities must be a non-empty two-dimensional matrix with at least two classes.")
    return _normalize_probability_rows(matrix, epsilon=epsilon)


def _normalize_probability_rows(values: np.ndarray, *, epsilon: float) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("probability rows must be finite and non-negative.")
    matrix = np.maximum(matrix, float(epsilon))
    row_sums = np.sum(matrix, axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("probability rows must have positive mass.")
    return matrix / row_sums


def _metadata(cfg: TargetPriorAdjustmentConfig, *, n_rows: int, n_classes: int, n_iter: int, converged: bool) -> dict[str, Any]:
    return {
        "target_prior_adjustment": True,
        "target_prior_adjustment_protocol": TARGET_PRIOR_ADJUSTMENT_PROTOCOL,
        "target_prior_adjustment_protocol_category": TARGET_PRIOR_ADJUSTMENT_CATEGORY,
        "target_prior_adjustment_estimator": cfg.estimator,
        "target_prior_adjustment_uses_target_probabilities": True,
        "target_prior_adjustment_uses_target_features": False,
        "target_prior_adjustment_uses_target_labels": False,
        "target_prior_adjustment_valid_for_strict_source_only": False,
        "target_prior_adjustment_valid_for_unlabeled_target_adaptation": True,
        "target_prior_adjustment_valid_for_benchmark": False,
        "target_prior_adjustment_n_rows": int(n_rows),
        "target_prior_adjustment_n_classes": int(n_classes),
        "target_prior_adjustment_strength": float(cfg.strength),
        "target_prior_adjustment_max_iter": int(cfg.max_iter),
        "target_prior_adjustment_n_iter": int(n_iter),
        "target_prior_adjustment_converged": bool(converged),
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


def _unit_interval_float(value: float | str, *, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return parsed
