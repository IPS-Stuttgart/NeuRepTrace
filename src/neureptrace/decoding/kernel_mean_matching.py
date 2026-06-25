"""Kernel mean matching for unlabeled target-adaptive source weighting.

Kernel mean matching (KMM) reweights source rows so the weighted source feature
distribution better matches an unlabeled held-out target feature distribution in
an RKHS.  The implementation here is deliberately dependency-light: it uses
NumPy plus SciPy's SLSQP optimizer rather than an external quadratic-programming
package.

The public API is Category 2 in the cross-subject protocol taxonomy.  It uses
source features and unlabeled target features, and may optionally use source
labels only for class-balanced post-normalization.  It does not accept held-out
target labels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize

KMM_PROTOCOL = "unlabeled_target_kernel_mean_matching"
KMM_PROTOCOL_CATEGORY = "2_unlabeled_target_adaptive"
KMM_KERNELS = ("rbf", "linear")
DEFAULT_KMM_MAX_WEIGHT = 10.0
DEFAULT_KMM_EPSILON = "auto"
DEFAULT_KMM_GAMMA = "median"
DEFAULT_KMM_REGULARIZATION = 1.0e-6
DEFAULT_KMM_MAX_ITER = 500
DEFAULT_KMM_TOL = 1.0e-8
_MIN_SCALE = 1.0e-12


@dataclass(frozen=True, slots=True)
class KernelMeanMatchingConfig:
    """Configuration for dependency-light KMM source weighting."""

    kernel: str = "rbf"
    gamma: float | str = DEFAULT_KMM_GAMMA
    max_weight: float = DEFAULT_KMM_MAX_WEIGHT
    epsilon: float | str | None = DEFAULT_KMM_EPSILON
    regularization: float = DEFAULT_KMM_REGULARIZATION
    max_iter: int = DEFAULT_KMM_MAX_ITER
    tol: float = DEFAULT_KMM_TOL
    normalize: bool = True
    class_balance: bool = False


@dataclass(frozen=True, slots=True)
class KernelMeanMatchingResult:
    """KMM source weights and provenance metadata."""

    weights: np.ndarray
    objective_value: float
    success: bool
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments,too-many-locals

def kernel_mean_matching_weights(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    target_features: Sequence[Sequence[float]] | np.ndarray,
    *,
    kernel: str | None = "rbf",
    gamma: float | str = DEFAULT_KMM_GAMMA,
    max_weight: float | str = DEFAULT_KMM_MAX_WEIGHT,
    epsilon: float | str | None = DEFAULT_KMM_EPSILON,
    regularization: float | str = DEFAULT_KMM_REGULARIZATION,
    max_iter: int | str = DEFAULT_KMM_MAX_ITER,
    tol: float | str = DEFAULT_KMM_TOL,
    normalize: bool = True,
    source_labels: Sequence[Any] | np.ndarray | None = None,
    class_balance: bool = False,
) -> KernelMeanMatchingResult:
    """Estimate source-row weights from unlabeled target features.

    Parameters
    ----------
    source_features:
        Source feature matrix, usually pooled over source subjects in an outer
        LOSO fold.
    target_features:
        Unlabeled held-out target feature matrix used for covariate-shift
        weighting.  The function intentionally has no ``target_labels`` argument.
    kernel:
        ``"rbf"`` or ``"linear"``.
    gamma:
        RBF kernel width.  ``"median"``/``"auto"`` uses the median pairwise
        squared distance over source and target rows.  ``"scale"`` uses
        ``1 / n_features``.
    max_weight:
        Upper bound for each source weight before optional normalization.
    epsilon:
        KMM sum-constraint slack.  ``"auto"`` uses the common
        ``(sqrt(n_source)-1)/sqrt(n_source)`` heuristic.  ``None`` disables the
        sum constraint and only enforces box bounds.
    regularization:
        Diagonal ridge added to the source-source kernel matrix.
    max_iter, tol:
        SLSQP optimizer controls.
    normalize:
        If true, final weights are rescaled to have mean one.
    source_labels:
        Optional source labels used only when ``class_balance=True``.
    class_balance:
        If true, rescale weights so each source class receives equal total mass.
        This uses source labels only and remains Category 2.

    Returns
    -------
    KernelMeanMatchingResult
        Mean-one source sample weights and protocol metadata.
    """

    source = _feature_matrix(source_features, name="source_features")
    target = _feature_matrix(target_features, name="target_features")
    if source.shape[1] != target.shape[1]:
        raise ValueError(f"source_features and target_features must have the same feature width: {source.shape[1]} != {target.shape[1]}.")
    normalized_kernel = normalize_kmm_kernel(kernel)
    max_weight_value = _positive_float(max_weight, name="max_weight")
    regularization_value = _nonnegative_float(regularization, name="regularization")
    max_iterations = _positive_int(max_iter, name="max_iter")
    tolerance = _positive_float(tol, name="tol")
    epsilon_value = normalize_kmm_epsilon(epsilon, n_source=source.shape[0])
    gamma_value = resolve_kmm_gamma(gamma, source, target) if normalized_kernel == "rbf" else ""

    source_kernel = _source_kernel(source, normalized_kernel, gamma_value)
    if regularization_value > 0.0:
        source_kernel = source_kernel + regularization_value * np.eye(source.shape[0], dtype=float)
    kappa = _kappa_vector(source, target, normalized_kernel, gamma_value)
    initial = np.ones(source.shape[0], dtype=float)
    bounds = [(0.0, max_weight_value) for _ in range(source.shape[0])]
    constraints = _sum_constraints(source.shape[0], epsilon_value)

    def objective(weights: np.ndarray) -> float:
        return float(0.5 * weights @ source_kernel @ weights - kappa @ weights)

    def gradient(weights: np.ndarray) -> np.ndarray:
        return source_kernel @ weights - kappa

    solution = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": max_iterations, "ftol": tolerance, "disp": False},
    )
    raw_weights = np.asarray(solution.x, dtype=float) if np.all(np.isfinite(solution.x)) else initial.copy()
    raw_weights = _project_weights(raw_weights, max_weight=max_weight_value, epsilon=epsilon_value, n_source=source.shape[0])
    status = str(solution.message)
    success = bool(solution.success)

    if not success:
        fallback = minimize(
            objective,
            initial,
            jac=gradient,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iterations, "ftol": tolerance},
        )
        if np.all(np.isfinite(fallback.x)) and objective(np.asarray(fallback.x, dtype=float)) <= objective(raw_weights) + 1.0e-10:
            raw_weights = _project_weights(np.asarray(fallback.x, dtype=float), max_weight=max_weight_value, epsilon=epsilon_value, n_source=source.shape[0])
            status = f"SLSQP failed ({status}); used bounded L-BFGS-B fallback: {fallback.message}"
            success = bool(fallback.success)

    weights = raw_weights.copy()
    if class_balance:
        if source_labels is None:
            raise ValueError("source_labels are required when class_balance=True.")
        weights = _class_balanced_weights(weights, source_labels)
    if normalize:
        weights = _mean_one_weights(weights)
    objective_value = objective(raw_weights)
    metadata = _metadata(
        n_source=source.shape[0],
        n_target=target.shape[0],
        feature_dim=source.shape[1],
        kernel=normalized_kernel,
        gamma=gamma_value,
        max_weight=max_weight_value,
        epsilon=epsilon_value,
        regularization=regularization_value,
        max_iter=max_iterations,
        tol=tolerance,
        normalize=bool(normalize),
        class_balance=bool(class_balance),
        success=success,
        status=status,
        objective_value=objective_value,
        weights=weights,
    )
    return KernelMeanMatchingResult(
        weights=weights.astype(np.float64, copy=False),
        objective_value=float(objective_value),
        success=success,
        status=status,
        metadata=metadata,
    )


def kmm_config(
    *,
    kernel: str | None = "rbf",
    gamma: float | str = DEFAULT_KMM_GAMMA,
    max_weight: float | str = DEFAULT_KMM_MAX_WEIGHT,
    epsilon: float | str | None = DEFAULT_KMM_EPSILON,
    regularization: float | str = DEFAULT_KMM_REGULARIZATION,
    max_iter: int | str = DEFAULT_KMM_MAX_ITER,
    tol: float | str = DEFAULT_KMM_TOL,
    normalize: bool = True,
    class_balance: bool = False,
) -> KernelMeanMatchingConfig:
    """Normalize user-facing KMM configuration values."""

    return KernelMeanMatchingConfig(
        kernel=normalize_kmm_kernel(kernel),
        gamma=gamma,
        max_weight=_positive_float(max_weight, name="max_weight"),
        epsilon=epsilon,
        regularization=_nonnegative_float(regularization, name="regularization"),
        max_iter=_positive_int(max_iter, name="max_iter"),
        tol=_positive_float(tol, name="tol"),
        normalize=bool(normalize),
        class_balance=bool(class_balance),
    )


def kernel_mean_matching_weights_from_config(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    target_features: Sequence[Sequence[float]] | np.ndarray,
    config: KernelMeanMatchingConfig | Mapping[str, Any] | None = None,
    *,
    source_labels: Sequence[Any] | np.ndarray | None = None,
) -> KernelMeanMatchingResult:
    """Estimate KMM weights from a dataclass or mapping configuration."""

    cfg = kmm_config() if config is None else config
    if isinstance(cfg, Mapping):
        cfg = kmm_config(**dict(cfg))
    return kernel_mean_matching_weights(
        source_features,
        target_features,
        kernel=cfg.kernel,
        gamma=cfg.gamma,
        max_weight=cfg.max_weight,
        epsilon=cfg.epsilon,
        regularization=cfg.regularization,
        max_iter=cfg.max_iter,
        tol=cfg.tol,
        normalize=cfg.normalize,
        source_labels=source_labels,
        class_balance=cfg.class_balance,
    )


def normalize_kmm_kernel(value: str | None) -> str:
    """Normalize supported KMM kernel aliases."""

    normalized = "rbf" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {
        "gaussian": "rbf",
        "gaussian_rbf": "rbf",
        "radial_basis": "rbf",
        "dot": "linear",
        "linear_kernel": "linear",
    }.get(normalized, normalized)
    if normalized not in KMM_KERNELS:
        raise ValueError(f"Unknown KMM kernel {value!r}. Available kernels: {', '.join(KMM_KERNELS)}.")
    return normalized


def resolve_kmm_gamma(value: float | str, source_features: Sequence[Sequence[float]] | np.ndarray, target_features: Sequence[Sequence[float]] | np.ndarray) -> float:
    """Resolve an RBF gamma value from a numeric value or heuristic alias."""

    source = _feature_matrix(source_features, name="source_features")
    target = _feature_matrix(target_features, name="target_features")
    if source.shape[1] != target.shape[1]:
        raise ValueError("source_features and target_features must have matching feature width.")
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_")
        if normalized in {"median", "auto", "median_distance", "median_heuristic"}:
            squared = _upper_pairwise_squared_distances(np.vstack([source, target]))
            positive = squared[squared > _MIN_SCALE]
            sigma2 = float(np.median(positive)) if positive.size else 1.0
            return 1.0 / (2.0 * max(sigma2, _MIN_SCALE))
        if normalized == "scale":
            return 1.0 / max(1, source.shape[1])
        numeric = float(normalized)
    else:
        numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError("gamma must be positive and finite, or one of: median, auto, scale.")
    return numeric


def normalize_kmm_epsilon(value: float | str | None, *, n_source: int) -> float | None:
    """Resolve KMM sum-constraint slack."""

    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_")
        if normalized in {"", "none", "off", "disabled"}:
            return None
        if normalized in {"auto", "default"}:
            n = max(1, int(n_source))
            return float((np.sqrt(n) - 1.0) / np.sqrt(n))
        numeric = float(normalized)
    else:
        numeric = float(value)
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError("epsilon must be finite and non-negative, 'auto', or None.")
    return numeric


def _source_kernel(source: np.ndarray, kernel: str, gamma: float | str) -> np.ndarray:
    if kernel == "linear":
        return source @ source.T
    return _rbf_kernel(source, source, float(gamma))


def _kappa_vector(source: np.ndarray, target: np.ndarray, kernel: str, gamma: float | str) -> np.ndarray:
    if kernel == "linear":
        cross = source @ target.T
    else:
        cross = _rbf_kernel(source, target, float(gamma))
    return (float(source.shape[0]) / float(target.shape[0])) * np.sum(cross, axis=1)


def _rbf_kernel(left: np.ndarray, right: np.ndarray, gamma: float) -> np.ndarray:
    return np.exp(-float(gamma) * _squared_euclidean(left, right))


def _squared_euclidean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.sum(left * left, axis=1, keepdims=True)
    right_norm = np.sum(right * right, axis=1, keepdims=True).T
    return np.maximum(left_norm + right_norm - 2.0 * (left @ right.T), 0.0)


def _upper_pairwise_squared_distances(values: np.ndarray) -> np.ndarray:
    if values.shape[0] <= 1:
        return np.asarray([1.0], dtype=float)
    squared = _squared_euclidean(values, values)
    return squared[np.triu_indices(values.shape[0], k=1)]


def _sum_constraints(n_source: int, epsilon: float | None) -> list[dict[str, Any]]:
    if epsilon is None:
        return []
    lower = float(n_source) * (1.0 - float(epsilon))
    upper = float(n_source) * (1.0 + float(epsilon))
    return [
        {"type": "ineq", "fun": lambda weights, lower=lower: np.sum(weights) - lower, "jac": lambda weights: np.ones_like(weights)},
        {"type": "ineq", "fun": lambda weights, upper=upper: upper - np.sum(weights), "jac": lambda weights: -np.ones_like(weights)},
    ]


def _project_weights(weights: np.ndarray, *, max_weight: float, epsilon: float | None, n_source: int) -> np.ndarray:
    projected = np.clip(np.asarray(weights, dtype=float).reshape(-1), 0.0, float(max_weight))
    if epsilon is None:
        return projected
    lower = float(n_source) * (1.0 - float(epsilon))
    upper = float(n_source) * (1.0 + float(epsilon))
    total = float(np.sum(projected))
    if total <= 0.0:
        projected[:] = min(1.0, float(max_weight))
        total = float(np.sum(projected))
    if total < lower:
        projected *= lower / max(total, _MIN_SCALE)
    elif total > upper:
        projected *= upper / max(total, _MIN_SCALE)
    return np.clip(projected, 0.0, float(max_weight))


def _class_balanced_weights(weights: np.ndarray, labels: Sequence[Any] | np.ndarray) -> np.ndarray:
    label_vector = np.asarray(labels, dtype=object).reshape(-1)
    if label_vector.shape[0] != weights.shape[0]:
        raise ValueError(f"source_labels must contain one label per source row: {label_vector.shape[0]} != {weights.shape[0]}.")
    balanced = np.asarray(weights, dtype=float).copy()
    classes = tuple(dict.fromkeys(label_vector.tolist()))
    if not classes:
        return balanced
    total = float(np.sum(balanced))
    if total <= 0.0:
        balanced[:] = 1.0
        total = float(np.sum(balanced))
    target_mass = total / float(len(classes))
    for class_label in classes:
        mask = label_vector == class_label
        class_mass = float(np.sum(balanced[mask]))
        if class_mass > 0.0:
            balanced[mask] *= target_mass / class_mass
    return balanced


def _mean_one_weights(weights: np.ndarray) -> np.ndarray:
    values = np.asarray(weights, dtype=float).reshape(-1).copy()
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("KMM weights must be finite and non-negative.")
    mean = float(np.mean(values))
    if mean <= 0.0:
        values[:] = 1.0
        return values
    return values / mean


def _metadata(
    *,
    n_source: int,
    n_target: int,
    feature_dim: int,
    kernel: str,
    gamma: float | str,
    max_weight: float,
    epsilon: float | None,
    regularization: float,
    max_iter: int,
    tol: float,
    normalize: bool,
    class_balance: bool,
    success: bool,
    status: str,
    objective_value: float,
    weights: np.ndarray,
) -> dict[str, Any]:
    return {
        "kernel_mean_matching": True,
        "kmm_protocol": KMM_PROTOCOL,
        "kmm_protocol_category": KMM_PROTOCOL_CATEGORY,
        "kmm_uses_source_features": True,
        "kmm_uses_source_labels": bool(class_balance),
        "kmm_uses_target_features": True,
        "kmm_uses_target_labels": False,
        "kmm_valid_for_strict_source_only": False,
        "kmm_valid_for_unlabeled_target_adaptation": True,
        "kmm_valid_for_target_calibration": False,
        "kmm_debug_upper_bound": False,
        "kmm_n_source_rows": int(n_source),
        "kmm_n_target_rows": int(n_target),
        "kmm_feature_dim": int(feature_dim),
        "kmm_kernel": kernel,
        "kmm_gamma": "" if gamma == "" else float(gamma),
        "kmm_max_weight": float(max_weight),
        "kmm_epsilon": "" if epsilon is None else float(epsilon),
        "kmm_regularization": float(regularization),
        "kmm_max_iter": int(max_iter),
        "kmm_tol": float(tol),
        "kmm_normalize": bool(normalize),
        "kmm_class_balance": bool(class_balance),
        "kmm_success": bool(success),
        "kmm_status": str(status),
        "kmm_objective_value": float(objective_value),
        "kmm_weight_min": float(np.min(weights)),
        "kmm_weight_mean": float(np.mean(weights)),
        "kmm_weight_max": float(np.max(weights)),
        "kmm_weight_std": float(np.std(weights)),
    }


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    if matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one row and one feature column.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _positive_int(value: int | str, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(parsed)


def _positive_float(value: float | str, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be positive and finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite.") from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return float(parsed)


def _nonnegative_float(value: float | str, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and non-negative.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and non-negative.") from exc
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return float(parsed)
