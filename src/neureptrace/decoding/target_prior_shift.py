"""Unlabeled target prior-shift probability adaptation.

This module adapts source-model class probabilities to an unlabeled target batch by
estimating target class priors with the classic EM update for prior-probability
shift.  The public API consumes probability rows and optional source priors; it
never accepts held-out target labels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

TARGET_PRIOR_SHIFT_PROTOCOL = "unlabeled_target_prior_shift_adaptation"
TARGET_PRIOR_SHIFT_CATEGORY = "2_unlabeled_target_adaptive"
DEFAULT_MAX_ITER = 200
DEFAULT_TOL = 1e-8
DEFAULT_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class TargetPriorShiftConfig:
    """Configuration for unlabeled target prior adaptation."""

    max_iter: int = DEFAULT_MAX_ITER
    tol: float = DEFAULT_TOL
    epsilon: float = DEFAULT_EPSILON
    initial_prior: str = "mean_probability"

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_iter", _positive_int(self.max_iter, name="max_iter"))
        object.__setattr__(self, "tol", _positive_float(self.tol, name="tol"))
        object.__setattr__(self, "epsilon", _positive_float(self.epsilon, name="epsilon"))
        object.__setattr__(self, "initial_prior", normalize_initial_prior(self.initial_prior))


@dataclass(frozen=True, slots=True)
class TargetPriorShiftResult:
    """Prior-adapted target probabilities and provenance metadata."""

    probabilities: np.ndarray
    original_probabilities: np.ndarray
    source_prior: np.ndarray
    estimated_target_prior: np.ndarray
    prior_ratio: np.ndarray
    n_iter: int
    converged: bool
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-locals

def adapt_target_probabilities_prior_shift(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    source_prior: Sequence[float] | np.ndarray | None = None,
    config: TargetPriorShiftConfig | Mapping[str, Any] | None = None,
) -> TargetPriorShiftResult:
    """Estimate target priors and adapt probability rows without target labels.

    Parameters
    ----------
    probabilities:
        Target-batch class-probability rows from a source-trained model.  Rows are
        normalized internally.
    source_prior:
        Optional source class prior in the same class order as the columns of
        ``probabilities``.  If omitted, a uniform source prior is used.
    config:
        EM settings.  A mapping is normalized through
        :func:`target_prior_shift_config`.
    """

    cfg = target_prior_shift_config() if config is None else _coerce_config(config)
    original = _probability_matrix(probabilities, name="probabilities", epsilon=cfg.epsilon)
    n_rows, n_classes = original.shape
    src_prior = _prior_vector(source_prior, n_classes=n_classes, epsilon=cfg.epsilon)
    target_prior = _initial_target_prior(original, source_prior=src_prior, mode=cfg.initial_prior, epsilon=cfg.epsilon)
    converged = False
    adapted = original.copy()
    for iteration in range(1, cfg.max_iter + 1):
        ratio = target_prior / src_prior
        adapted = _normalize_probability_rows(original * ratio[None, :], epsilon=cfg.epsilon)
        updated_prior = _normalize_probability_rows(np.mean(adapted, axis=0, keepdims=True), epsilon=cfg.epsilon)[0]
        delta = float(np.max(np.abs(updated_prior - target_prior)))
        target_prior = updated_prior
        if delta <= cfg.tol:
            converged = True
            break
    else:
        iteration = cfg.max_iter
    final_ratio = target_prior / src_prior
    adapted = _normalize_probability_rows(original * final_ratio[None, :], epsilon=cfg.epsilon)
    metadata = {
        "target_prior_shift": True,
        "target_prior_shift_protocol": TARGET_PRIOR_SHIFT_PROTOCOL,
        "target_prior_shift_protocol_category": TARGET_PRIOR_SHIFT_CATEGORY,
        "target_prior_shift_uses_target_probabilities": True,
        "target_prior_shift_uses_target_features": False,
        "target_prior_shift_uses_target_labels": False,
        "target_prior_shift_valid_for_strict_source_only": False,
        "target_prior_shift_valid_for_unlabeled_target_adaptation": True,
        "target_prior_shift_valid_for_benchmark": False,
        "target_prior_shift_n_rows": int(n_rows),
        "target_prior_shift_n_classes": int(n_classes),
        "target_prior_shift_max_iter": int(cfg.max_iter),
        "target_prior_shift_n_iter": int(iteration),
        "target_prior_shift_converged": bool(converged),
        "target_prior_shift_tol": float(cfg.tol),
        "target_prior_shift_epsilon": float(cfg.epsilon),
        "target_prior_shift_initial_prior": cfg.initial_prior,
        "target_prior_shift_source_prior": "|".join(f"{value:.12g}" for value in src_prior.tolist()),
        "target_prior_shift_estimated_target_prior": "|".join(f"{value:.12g}" for value in target_prior.tolist()),
    }
    return TargetPriorShiftResult(
        probabilities=adapted.astype(np.float32, copy=False),
        original_probabilities=original.astype(np.float32, copy=False),
        source_prior=src_prior.astype(np.float32, copy=False),
        estimated_target_prior=target_prior.astype(np.float32, copy=False),
        prior_ratio=final_ratio.astype(np.float32, copy=False),
        n_iter=int(iteration),
        converged=bool(converged),
        metadata=metadata,
    )


def target_prior_shift_config(
    *,
    max_iter: int | str = DEFAULT_MAX_ITER,
    tol: float | str = DEFAULT_TOL,
    epsilon: float | str = DEFAULT_EPSILON,
    initial_prior: str | None = "mean_probability",
) -> TargetPriorShiftConfig:
    """Normalize target-prior adaptation options."""

    return TargetPriorShiftConfig(
        max_iter=_positive_int(max_iter, name="max_iter"),
        tol=_positive_float(tol, name="tol"),
        epsilon=_positive_float(epsilon, name="epsilon"),
        initial_prior=normalize_initial_prior(initial_prior),
    )


def normalize_initial_prior(value: str | None) -> str:
    """Normalize initial target-prior aliases."""

    normalized = "mean_probability" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {
        "mean": "mean_probability",
        "average_probability": "mean_probability",
        "source": "source_prior",
        "uniform": "uniform",
    }.get(normalized, normalized)
    if normalized not in {"mean_probability", "source_prior", "uniform"}:
        raise ValueError("initial_prior must be 'mean_probability', 'source_prior', or 'uniform'.")
    return normalized


def _coerce_config(config: TargetPriorShiftConfig | Mapping[str, Any]) -> TargetPriorShiftConfig:
    if isinstance(config, TargetPriorShiftConfig):
        return config
    return target_prior_shift_config(**dict(config))


def _initial_target_prior(probabilities: np.ndarray, *, source_prior: np.ndarray, mode: str, epsilon: float) -> np.ndarray:
    if mode == "mean_probability":
        return _normalize_probability_rows(np.mean(probabilities, axis=0, keepdims=True), epsilon=epsilon)[0]
    if mode == "source_prior":
        return source_prior.copy()
    if mode == "uniform":
        return np.full(probabilities.shape[1], 1.0 / probabilities.shape[1], dtype=float)
    raise ValueError(f"Unhandled initial_prior {mode!r}.")


def _probability_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str, epsilon: float) -> np.ndarray:
    materialized = _materialize_iterables(values)
    _reject_boolean_values(materialized, name=name)
    matrix = np.asarray(materialized, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 2:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix with at least two columns.")
    return _normalize_probability_rows(matrix, epsilon=epsilon)


def _prior_vector(values: Sequence[float] | np.ndarray | None, *, n_classes: int, epsilon: float) -> np.ndarray:
    if values is None:
        vector = np.full(n_classes, 1.0 / n_classes, dtype=float)
    else:
        materialized = _materialize_iterables(values)
        _reject_boolean_values(materialized, name="source_prior")
        vector = np.asarray(materialized, dtype=float).reshape(-1)
        if vector.shape[0] != n_classes:
            raise ValueError(f"source_prior must contain one value per probability column: {vector.shape[0]} != {n_classes}.")
    return _normalize_probability_rows(vector[None, :], epsilon=epsilon)[0]


def _normalize_probability_rows(values: np.ndarray, *, epsilon: float) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("probabilities must be finite and non-negative.")
    matrix = np.maximum(matrix, float(epsilon))
    row_sums = np.sum(matrix, axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("probability rows must have positive mass.")
    return matrix / row_sums


def _materialize_iterables(values: Any) -> Any:
    """Recursively materialize one-pass iterable probability/prior inputs."""

    if isinstance(values, np.ndarray) or isinstance(values, (str, bytes)):
        return values
    try:
        iterator = iter(values)
    except TypeError:
        return values
    return [_materialize_iterables(item) for item in iterator]


def _reject_boolean_values(values: Any, *, name: str) -> None:
    if _contains_boolean_value(values):
        raise ValueError(f"{name} must not contain boolean values.")


def _contains_boolean_value(values: Any) -> bool:
    if isinstance(values, (bool, np.bool_)):
        return True
    if isinstance(values, np.ndarray):
        if np.issubdtype(values.dtype, np.bool_):
            return True
        if values.dtype == object:
            return any(_contains_boolean_value(item) for item in values.flat)
        return False
    if isinstance(values, (str, bytes)):
        return False
    try:
        iterator = iter(values)
    except TypeError:
        return False
    return any(_contains_boolean_value(item) for item in iterator)


def _numeric_scalar(value: object, *, name: str, expectation: str) -> float:
    message = f"{name} must be {expectation}."
    if isinstance(value, (bool, np.bool_)) or isinstance(value, np.ndarray):
        raise ValueError(message)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc


def _positive_int(value: int | str, *, name: str) -> int:
    parsed = _numeric_scalar(value, name=name, expectation="a positive integer")
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(parsed)


def _positive_float(value: float | str, *, name: str) -> float:
    parsed = _numeric_scalar(value, name=name, expectation="positive and finite")
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed
