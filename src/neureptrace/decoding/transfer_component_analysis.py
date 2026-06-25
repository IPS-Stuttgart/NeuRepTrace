"""Transfer Component Analysis for unlabeled target-adaptive decoding.

Transfer Component Analysis (TCA) learns a shared latent space by reducing the
marginal distribution discrepancy between labeled source rows and unlabeled target
rows.  The downstream probe is trained only with source labels in that latent
space.  The public APIs in this module intentionally do not accept target labels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression

TCA_PROTOCOL = "unlabeled_target_transfer_component_analysis"
TCA_CATEGORY = "2_unlabeled_target_adaptive"
TCA_KERNELS = ("linear", "rbf")
DEFAULT_TCA_COMPONENTS = 16
DEFAULT_TCA_REGULARIZATION = 1.0
DEFAULT_TCA_EPSILON = 1e-8
DEFAULT_TCA_GAMMA = "median"


@dataclass(frozen=True, slots=True)
class TransferComponentAnalysisModel:
    """Fitted TCA projection and provenance."""

    source_fit_features: np.ndarray
    target_fit_features: np.ndarray
    fit_features_standardized: np.ndarray
    projection: np.ndarray
    kernel: str
    gamma: float | None
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    latent_mean: np.ndarray
    latent_scale: np.ndarray
    n_components: int
    regularization: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TransferComponentAnalysisResult:
    """Source/target latent features and fitted TCA model."""

    source_features: np.ndarray
    target_features: np.ndarray
    model: TransferComponentAnalysisModel
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TCATransferClassificationResult:
    """Source-label classifier outputs after TCA feature transfer."""

    source_features: np.ndarray
    target_features: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray | None
    classes: np.ndarray
    model: TransferComponentAnalysisModel
    classifier: BaseEstimator
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments,too-many-locals

def transfer_component_analysis_features(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    target_features: Sequence[Sequence[float]] | np.ndarray,
    *,
    n_components: int | str | None = DEFAULT_TCA_COMPONENTS,
    kernel: str | None = "linear",
    regularization: float | str = DEFAULT_TCA_REGULARIZATION,
    gamma: float | str | None = DEFAULT_TCA_GAMMA,
    standardize: bool = True,
    normalize_components: bool = True,
    epsilon: float | str = DEFAULT_TCA_EPSILON,
) -> TransferComponentAnalysisResult:
    """Fit TCA and return source/target latent features.

    Parameters
    ----------
    source_features:
        Labeled source feature rows.  Source labels are not needed by TCA itself.
    target_features:
        Unlabeled held-out target feature rows used to estimate the target feature
        distribution.  This makes the method Category 2 rather than strict
        source-only.
    n_components:
        Requested latent dimensionality.  The effective dimensionality is capped
        by the number of source plus target rows.
    kernel:
        ``"linear"`` or ``"rbf"``.
    regularization:
        Positive ridge term in the generalized eigenproblem.
    gamma:
        RBF gamma.  ``"median"`` uses the median pairwise squared distance among
        source plus target fit rows.  Ignored for the linear kernel.
    standardize:
        If true, z-score features using source-plus-target feature statistics.
        This is still Category 2 because unlabeled target features contribute.
    normalize_components:
        If true, z-score latent TCA components using source-plus-target latent
        statistics from the fit set.
    epsilon:
        Numerical floor for standard deviations and generalized-eigenproblem
        regularization.

    Returns
    -------
    TransferComponentAnalysisResult
        Source and target latent features plus a reusable fitted model.

    Notes
    -----
    This function intentionally has no ``target_labels`` argument.  Source labels
    should be used only by the downstream classifier.
    """

    source = _feature_matrix(source_features, name="source_features")
    target = _feature_matrix(target_features, name="target_features")
    if source.shape[1] != target.shape[1]:
        raise ValueError(f"source_features and target_features must have the same feature width: {source.shape[1]} != {target.shape[1]}.")
    if source.shape[0] < 1 or target.shape[0] < 1:
        raise ValueError("TCA requires at least one source row and one target row.")

    kernel_name = normalize_tca_kernel(kernel)
    reg = _positive_float(regularization, name="regularization")
    eps = _positive_float(epsilon, name="epsilon")
    combined = np.vstack([source, target])
    standardized, feature_mean, feature_scale = _standardize_fit_features(combined, enabled=bool(standardize), epsilon=eps)
    resolved_gamma = _resolve_gamma(standardized, gamma=gamma, kernel=kernel_name, epsilon=eps)
    k_matrix = _kernel_matrix(standardized, standardized, kernel=kernel_name, gamma=resolved_gamma)
    n_total = combined.shape[0]
    n_source = source.shape[0]
    n_target = target.shape[0]
    n_components_resolved = _effective_components(n_components, max_components=n_total)

    mmd_matrix = _mmd_matrix(n_source, n_target)
    centering = np.eye(n_total, dtype=float) - np.full((n_total, n_total), 1.0 / n_total, dtype=float)
    left = k_matrix @ mmd_matrix @ k_matrix + reg * np.eye(n_total, dtype=float)
    right = k_matrix @ centering @ k_matrix + eps * np.eye(n_total, dtype=float)
    eigenvalues, eigenvectors = eigh(_symmetrize(left), _symmetrize(right), check_finite=True)
    order = np.argsort(eigenvalues)
    projection = eigenvectors[:, order[:n_components_resolved]]
    latent = k_matrix @ projection
    latent, latent_mean, latent_scale = _normalize_latent(latent, enabled=bool(normalize_components), epsilon=eps)

    source_latent = latent[:n_source]
    target_latent = latent[n_source:]
    metadata = _metadata(
        n_source_rows=n_source,
        n_target_rows=n_target,
        feature_dim=source.shape[1],
        n_components=n_components_resolved,
        requested_components=n_components,
        kernel=kernel_name,
        gamma=resolved_gamma,
        regularization=reg,
        standardize=bool(standardize),
        normalize_components=bool(normalize_components),
        eigenvalues=eigenvalues[order[:n_components_resolved]],
        source_latent=source_latent,
        target_latent=target_latent,
    )
    model = TransferComponentAnalysisModel(
        source_fit_features=source.astype(np.float32, copy=False),
        target_fit_features=target.astype(np.float32, copy=False),
        fit_features_standardized=standardized.astype(np.float32, copy=False),
        projection=projection.astype(np.float32, copy=False),
        kernel=kernel_name,
        gamma=resolved_gamma,
        feature_mean=feature_mean.astype(np.float32, copy=False),
        feature_scale=feature_scale.astype(np.float32, copy=False),
        latent_mean=latent_mean.astype(np.float32, copy=False),
        latent_scale=latent_scale.astype(np.float32, copy=False),
        n_components=n_components_resolved,
        regularization=reg,
        metadata=metadata,
    )
    return TransferComponentAnalysisResult(
        source_features=source_latent.astype(np.float32, copy=False),
        target_features=target_latent.astype(np.float32, copy=False),
        model=model,
        metadata=metadata,
    )


def transform_with_tca_model(model: TransferComponentAnalysisModel, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Project new rows with an already fitted TCA model."""

    matrix = _feature_matrix(features, name="features")
    if matrix.shape[1] != model.feature_mean.shape[0]:
        raise ValueError(f"features width {matrix.shape[1]} does not match fitted width {model.feature_mean.shape[0]}.")
    standardized = (matrix - model.feature_mean) / model.feature_scale
    k_new = _kernel_matrix(standardized, model.fit_features_standardized, kernel=model.kernel, gamma=model.gamma)
    latent = k_new @ model.projection
    latent = (latent - model.latent_mean) / model.latent_scale
    return latent.astype(np.float32, copy=False)


# pylint: disable-next=too-many-arguments,too-many-locals

def fit_tca_transfer_classifier(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    target_features: Sequence[Sequence[float]] | np.ndarray,
    classifier: BaseEstimator | None = None,
    classifier_C: float | str = 1.0,
    classifier_max_iter: int | str = 1000,
    classifier_class_weight: str | Mapping[Any, float] | None = "balanced",
    sample_weight: Sequence[float] | np.ndarray | None = None,
    n_components: int | str | None = DEFAULT_TCA_COMPONENTS,
    kernel: str | None = "linear",
    regularization: float | str = DEFAULT_TCA_REGULARIZATION,
    gamma: float | str | None = DEFAULT_TCA_GAMMA,
    standardize: bool = True,
    normalize_components: bool = True,
    epsilon: float | str = DEFAULT_TCA_EPSILON,
) -> TCATransferClassificationResult:
    """Train a source-label classifier after Category-2 TCA alignment."""

    labels = np.asarray(source_labels, dtype=object).reshape(-1)
    source = _feature_matrix(source_features, name="source_features")
    if labels.shape[0] != source.shape[0]:
        raise ValueError(f"source_labels must contain one label per source row: {labels.shape[0]} != {source.shape[0]}.")
    if np.unique(labels).shape[0] < 2:
        raise ValueError("source_labels must contain at least two classes.")
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)
    if weights is not None:
        if weights.shape[0] != labels.shape[0]:
            raise ValueError(f"sample_weight must contain one value per source row: {weights.shape[0]} != {labels.shape[0]}.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("sample_weight must contain finite non-negative values.")

    tca = transfer_component_analysis_features(
        source,
        target_features,
        n_components=n_components,
        kernel=kernel,
        regularization=regularization,
        gamma=gamma,
        standardize=standardize,
        normalize_components=normalize_components,
        epsilon=epsilon,
    )
    model = clone(classifier) if classifier is not None else LogisticRegression(
        C=_positive_float(classifier_C, name="classifier_C"),
        max_iter=_positive_int(classifier_max_iter, name="classifier_max_iter"),
        class_weight=classifier_class_weight,
        random_state=13,
    )
    fit_kwargs = {} if weights is None else {"sample_weight": weights}
    model.fit(tca.source_features, labels, **fit_kwargs)
    predictions = np.asarray(model.predict(tca.target_features))
    probabilities = _predict_probabilities_or_none(model, tca.target_features)
    classes = np.asarray(getattr(model, "classes_", np.unique(labels)))
    metadata = {
        **tca.metadata,
        "tca_classifier": type(model).__name__,
        "tca_classifier_uses_source_labels": True,
        "tca_classifier_uses_target_labels": False,
        "tca_classifier_n_classes": int(classes.shape[0]),
    }
    return TCATransferClassificationResult(
        source_features=tca.source_features,
        target_features=tca.target_features,
        predictions=predictions,
        probabilities=probabilities,
        classes=classes,
        model=tca.model,
        classifier=model,
        metadata=metadata,
    )


def normalize_tca_kernel(kernel: str | None) -> str:
    """Normalize public aliases for TCA kernels."""

    normalized = "linear" if kernel is None else str(kernel).strip().lower().replace("-", "_")
    normalized = {"lin": "linear", "rbf_kernel": "rbf", "gaussian": "rbf", "gaussian_rbf": "rbf"}.get(normalized, normalized)
    if normalized not in TCA_KERNELS:
        raise ValueError(f"Unknown TCA kernel {kernel!r}. Available kernels: {', '.join(TCA_KERNELS)}.")
    return normalized


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    if matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one row and one feature column.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _standardize_fit_features(matrix: np.ndarray, *, enabled: bool, epsilon: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(matrix, axis=0)
    if not enabled:
        return matrix.astype(float, copy=True), np.zeros(matrix.shape[1], dtype=float), np.ones(matrix.shape[1], dtype=float)
    centered = matrix - mean
    scale = np.sqrt(np.maximum(np.var(centered, axis=0), epsilon))
    return centered / scale, mean, scale


def _normalize_latent(latent: np.ndarray, *, enabled: bool, epsilon: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(latent, axis=0)
    if not enabled:
        return latent.astype(float, copy=True), np.zeros(latent.shape[1], dtype=float), np.ones(latent.shape[1], dtype=float)
    centered = latent - mean
    scale = np.sqrt(np.maximum(np.var(centered, axis=0), epsilon))
    return centered / scale, mean, scale


def _mmd_matrix(n_source: int, n_target: int) -> np.ndarray:
    weights = np.concatenate([np.full(n_source, 1.0 / n_source), np.full(n_target, -1.0 / n_target)])
    matrix = np.outer(weights, weights)
    norm = float(np.linalg.norm(matrix, ord="fro"))
    return matrix if norm <= 0.0 else matrix / norm


def _kernel_matrix(left: np.ndarray, right: np.ndarray, *, kernel: str, gamma: float | None) -> np.ndarray:
    if kernel == "linear":
        return np.asarray(left @ right.T, dtype=float)
    if kernel == "rbf":
        if gamma is None:
            raise ValueError("RBF TCA requires a resolved gamma value.")
        return np.exp(-float(gamma) * _squared_euclidean(left, right))
    raise ValueError(f"Unhandled TCA kernel {kernel!r}.")


def _resolve_gamma(matrix: np.ndarray, *, gamma: float | str | None, kernel: str, epsilon: float) -> float | None:
    if kernel == "linear":
        return None
    if gamma is None:
        return 1.0 / max(matrix.shape[1], 1)
    if isinstance(gamma, str):
        normalized = gamma.strip().lower()
        if normalized in {"", "default", "auto", "median"}:
            median_distance = _median_positive_squared_distance(matrix)
            return 1.0 / max(2.0 * median_distance, epsilon)
        value = float(normalized)
    else:
        value = float(gamma)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("gamma must be positive and finite, 'median', or None.")
    return value


def _squared_euclidean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.sum(left * left, axis=1, keepdims=True)
    right_norm = np.sum(right * right, axis=1, keepdims=True).T
    return np.maximum(left_norm + right_norm - 2.0 * (left @ right.T), 0.0)


def _median_positive_squared_distance(matrix: np.ndarray) -> float:
    if matrix.shape[0] <= 1:
        return 1.0
    distances = _squared_euclidean(matrix, matrix)
    upper = distances[np.triu_indices(matrix.shape[0], k=1)]
    positive = upper[upper > 0.0]
    return 1.0 if positive.size == 0 else float(np.median(positive))


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _effective_components(value: int | str | None, *, max_components: int) -> int:
    if value is None:
        requested = DEFAULT_TCA_COMPONENTS
    elif isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "default"}:
            requested = DEFAULT_TCA_COMPONENTS
        elif text in {"all", "full", "inf", "infinity"}:
            return int(max_components)
        else:
            requested = float(text)
    else:
        requested = float(value)
    if not np.isfinite(requested) or requested % 1.0 != 0.0 or requested < 1.0:
        raise ValueError("n_components must be a positive integer, 'all', or infinity.")
    return min(int(requested), int(max_components))


def _metadata(
    *,
    n_source_rows: int,
    n_target_rows: int,
    feature_dim: int,
    n_components: int,
    requested_components: int | str | None,
    kernel: str,
    gamma: float | None,
    regularization: float,
    standardize: bool,
    normalize_components: bool,
    eigenvalues: np.ndarray,
    source_latent: np.ndarray,
    target_latent: np.ndarray,
) -> dict[str, Any]:
    source_mean = np.mean(source_latent, axis=0)
    target_mean = np.mean(target_latent, axis=0)
    latent_mean_distance = float(np.linalg.norm(source_mean - target_mean) / np.sqrt(max(1, n_components)))
    return {
        "tca": True,
        "tca_protocol": TCA_PROTOCOL,
        "tca_protocol_category": TCA_CATEGORY,
        "tca_uses_source_features": True,
        "tca_uses_target_features": True,
        "tca_uses_target_labels": False,
        "tca_valid_for_strict_source_only": False,
        "tca_valid_for_unlabeled_target_adaptation": True,
        "tca_valid_for_benchmark": False,
        "tca_n_source_rows": int(n_source_rows),
        "tca_n_target_rows": int(n_target_rows),
        "tca_feature_dim": int(feature_dim),
        "tca_n_components": int(n_components),
        "tca_requested_components": "" if requested_components is None else str(requested_components),
        "tca_kernel": kernel,
        "tca_gamma": "" if gamma is None else float(gamma),
        "tca_regularization": float(regularization),
        "tca_standardize": bool(standardize),
        "tca_normalize_components": bool(normalize_components),
        "tca_eigenvalues": "|".join(f"{float(value):.12g}" for value in np.asarray(eigenvalues).reshape(-1)),
        "tca_latent_source_target_mean_distance": latent_mean_distance,
    }


def _predict_probabilities_or_none(model: BaseEstimator, features: np.ndarray) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features), dtype=float)
        return _normalize_probability_rows(probabilities)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features), dtype=float)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(np.clip(shifted, -50.0, 50.0))
        return _normalize_probability_rows(exp_scores)
    return None


def _normalize_probability_rows(probabilities: np.ndarray) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("Predicted probabilities must be a finite two-dimensional array.")
    matrix = np.maximum(matrix, 0.0)
    row_sums = np.sum(matrix, axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Predicted probability rows must have positive mass.")
    return matrix / row_sums


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
    return parsed
