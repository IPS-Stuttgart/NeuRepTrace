"""PCA subspace alignment for unlabeled target-adaptive decoding.

This module implements a dependency-light version of feature-space subspace
alignment for cross-subject transfer.  A source PCA basis and an unlabeled target
PCA basis are estimated inside one fold, the source basis is rotated toward the
target basis, and the transformed source rows can be used with ordinary
source-label classifiers.  Target labels are intentionally absent from the public
API.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression

SUBSPACE_ALIGNMENT_PROTOCOL = "unlabeled_target_subspace_alignment"
SUBSPACE_ALIGNMENT_CATEGORY = "2_unlabeled_target_adaptive"
SUBSPACE_STANDARDIZATION_SCOPES = ("source", "source_target", "none")
DEFAULT_SUBSPACE_COMPONENTS = 32
_MIN_SCALE = 1.0e-12


@dataclass(frozen=True, slots=True)
class SubspaceAlignmentModel:
    """Fitted source-to-target PCA subspace alignment model."""

    source_mean: np.ndarray
    source_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    source_basis: np.ndarray
    target_basis: np.ndarray
    alignment_matrix: np.ndarray
    standardization_scope: str

    def transform_source(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        """Project source-domain rows into the target subspace coordinates."""

        matrix = _feature_matrix(features, name="features")
        if matrix.shape[1] != self.source_basis.shape[0]:
            raise ValueError(f"features width {matrix.shape[1]} does not match fitted width {self.source_basis.shape[0]}.")
        prepared = (matrix - self.source_mean) / self.source_scale
        return (prepared @ self.source_basis @ self.alignment_matrix).astype(np.float32, copy=False)

    def transform_target(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        """Project target-domain rows into their own PCA subspace coordinates."""

        matrix = _feature_matrix(features, name="features")
        if matrix.shape[1] != self.target_basis.shape[0]:
            raise ValueError(f"features width {matrix.shape[1]} does not match fitted width {self.target_basis.shape[0]}.")
        prepared = (matrix - self.target_mean) / self.target_scale
        return (prepared @ self.target_basis).astype(np.float32, copy=False)


@dataclass(frozen=True, slots=True)
class SubspaceAlignmentResult:
    """Aligned source and target features plus protocol metadata."""

    source_features: np.ndarray
    target_features: np.ndarray
    model: SubspaceAlignmentModel
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SubspaceAlignedClassificationResult:
    """Classifier outputs from a source-label probe in aligned subspace."""

    source_features: np.ndarray
    target_features: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray | None
    classes: np.ndarray
    classifier: BaseEstimator
    model: SubspaceAlignmentModel
    metadata: dict[str, Any] = field(default_factory=dict)


def fit_subspace_alignment(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    target_features: Sequence[Sequence[float]] | np.ndarray,
    *,
    n_components: int | str | float = DEFAULT_SUBSPACE_COMPONENTS,
    standardization_scope: str | None = "source",
) -> SubspaceAlignmentResult:
    """Fit Category-2 PCA subspace alignment from source and unlabeled target rows.

    Parameters
    ----------
    source_features, target_features:
        Source and held-out target feature rows.  Target rows are treated as
        unlabeled adaptation data.
    n_components:
        Requested latent dimensionality.  The effective value is capped by source
        rows, target rows, and feature width.
    standardization_scope:
        ``"source"`` standardizes both domains with source statistics before the
        separate PCA fits.  ``"source_target"`` standardizes with pooled source
        plus target statistics.  ``"none"`` only centers each domain.

    Returns
    -------
    SubspaceAlignmentResult
        Source rows in aligned target-subspace coordinates, target rows in target
        PCA coordinates, the fitted model, and Category-2 metadata.
    """

    source = _feature_matrix(source_features, name="source_features")
    target = _feature_matrix(target_features, name="target_features")
    if source.shape[1] != target.shape[1]:
        raise ValueError(f"source_features and target_features must have the same width: {source.shape[1]} != {target.shape[1]}.")
    scope = normalize_standardization_scope(standardization_scope)
    components = _effective_components(n_components, source_rows=source.shape[0], target_rows=target.shape[0], feature_dim=source.shape[1])
    source_mean, source_scale, target_mean, target_scale = _standardization(source, target, scope=scope)
    prepared_source = (source - source_mean) / source_scale
    prepared_target = (target - target_mean) / target_scale
    source_basis, source_variance = _pca_basis(prepared_source, components)
    target_basis, target_variance = _pca_basis(prepared_target, components)
    alignment = source_basis.T @ target_basis
    model = SubspaceAlignmentModel(
        source_mean=source_mean,
        source_scale=source_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        source_basis=source_basis,
        target_basis=target_basis,
        alignment_matrix=alignment,
        standardization_scope=scope,
    )
    aligned_source = model.transform_source(source)
    aligned_target = model.transform_target(target)
    metadata = _metadata(
        n_source_rows=source.shape[0],
        n_target_rows=target.shape[0],
        feature_dim=source.shape[1],
        n_components=components,
        requested_components=n_components,
        standardization_scope=scope,
        source_explained_variance=source_variance,
        target_explained_variance=target_variance,
    )
    return SubspaceAlignmentResult(source_features=aligned_source, target_features=aligned_target, model=model, metadata=metadata)


def fit_subspace_aligned_classifier(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    target_features: Sequence[Sequence[float]] | np.ndarray,
    n_components: int | str | float = DEFAULT_SUBSPACE_COMPONENTS,
    standardization_scope: str | None = "source",
    classifier: BaseEstimator | None = None,
    classifier_C: float = 1.0,
    classifier_max_iter: int = 1000,
    classifier_class_weight: str | Mapping[Any, float] | None = "balanced",
    sample_weight: Sequence[float] | np.ndarray | None = None,
) -> SubspaceAlignedClassificationResult:
    """Train a source-label classifier after Category-2 subspace alignment."""

    labels = _object_vector(source_labels, name="source_labels")
    aligned = fit_subspace_alignment(
        source_features,
        target_features,
        n_components=n_components,
        standardization_scope=standardization_scope,
    )
    if labels.shape[0] != aligned.source_features.shape[0]:
        raise ValueError(f"source_labels must contain one value per source row: {labels.shape[0]} != {aligned.source_features.shape[0]}.")
    if len(dict.fromkeys(labels.tolist())) < 2:
        raise ValueError("source_labels must contain at least two classes.")
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)
    if weights is not None:
        if weights.shape[0] != labels.shape[0]:
            raise ValueError(f"sample_weight must contain one value per source row: {weights.shape[0]} != {labels.shape[0]}.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("sample_weight must contain finite non-negative values.")
    model = clone(classifier) if classifier is not None else LogisticRegression(
        C=_positive_float(classifier_C, name="classifier_C"),
        class_weight=classifier_class_weight,
        max_iter=_positive_int(classifier_max_iter, name="classifier_max_iter"),
        random_state=13,
    )
    fit_kwargs = {} if weights is None else {"sample_weight": weights}
    model.fit(aligned.source_features, labels, **fit_kwargs)
    predictions = _predict_with_object_classes(model, aligned.target_features)
    probabilities = _probabilities_or_none(model, aligned.target_features)
    metadata = {
        **aligned.metadata,
        "subspace_alignment_classifier": type(model).__name__,
        "subspace_alignment_uses_source_labels": True,
        "subspace_alignment_uses_target_labels": False,
    }
    return SubspaceAlignedClassificationResult(
        source_features=aligned.source_features,
        target_features=aligned.target_features,
        predictions=predictions,
        probabilities=probabilities,
        classes=_classes_from_model(model, labels),
        classifier=model,
        model=aligned.model,
        metadata=metadata,
    )


def normalize_standardization_scope(value: str | None) -> str:
    """Normalize standardization-scope aliases."""

    normalized = "source" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {
        "train": "source",
        "source_only": "source",
        "source_stats": "source",
        "pooled": "source_target",
        "source_plus_target": "source_target",
        "source_and_target": "source_target",
        "target_adaptive": "source_target",
        "off": "none",
        "false": "none",
        "identity": "none",
    }.get(normalized, normalized)
    if normalized not in SUBSPACE_STANDARDIZATION_SCOPES:
        raise ValueError(f"Unknown standardization_scope {value!r}. Available scopes: {', '.join(SUBSPACE_STANDARDIZATION_SCOPES)}.")
    return normalized


def _metadata(*, n_source_rows: int, n_target_rows: int, feature_dim: int, n_components: int, requested_components: int | str | float, standardization_scope: str, source_explained_variance: np.ndarray, target_explained_variance: np.ndarray) -> dict[str, Any]:
    return {
        "subspace_alignment": True,
        "subspace_alignment_protocol": SUBSPACE_ALIGNMENT_PROTOCOL,
        "subspace_alignment_protocol_category": SUBSPACE_ALIGNMENT_CATEGORY,
        "subspace_alignment_uses_source_features": True,
        "subspace_alignment_uses_target_features": True,
        "subspace_alignment_uses_target_labels": False,
        "subspace_alignment_valid_for_strict_source_only": False,
        "subspace_alignment_valid_for_unlabeled_target_adaptation": True,
        "subspace_alignment_valid_for_target_calibration": False,
        "subspace_alignment_n_source_rows": int(n_source_rows),
        "subspace_alignment_n_target_rows": int(n_target_rows),
        "subspace_alignment_feature_dim": int(feature_dim),
        "subspace_alignment_n_components": int(n_components),
        "subspace_alignment_requested_components": str(requested_components),
        "subspace_alignment_standardization_scope": standardization_scope,
        "subspace_alignment_source_explained_variance": _format_vector(source_explained_variance),
        "subspace_alignment_target_explained_variance": _format_vector(target_explained_variance),
    }


def _standardization(source: np.ndarray, target: np.ndarray, *, scope: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if scope == "none":
        return np.mean(source, axis=0), np.ones(source.shape[1]), np.mean(target, axis=0), np.ones(target.shape[1])
    if scope == "source":
        mean = np.mean(source, axis=0)
        scale = _safe_std(source)
        return mean, scale, mean, scale
    pooled = np.vstack([source, target])
    mean = np.mean(pooled, axis=0)
    scale = _safe_std(pooled)
    return mean, scale, mean, scale


def _pca_basis(features: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    centered = features - np.mean(features, axis=0, keepdims=True)
    _u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:n_components].T
    total = float(np.sum(singular_values**2))
    explained = np.zeros(n_components, dtype=float) if total <= 0.0 else (singular_values[:n_components] ** 2) / total
    return _canonicalize_basis_signs(basis), explained


def _canonicalize_basis_signs(basis: np.ndarray) -> np.ndarray:
    fixed = np.asarray(basis, dtype=float).copy()
    for column in range(fixed.shape[1]):
        pivot = int(np.argmax(np.abs(fixed[:, column])))
        if fixed[pivot, column] < 0.0:
            fixed[:, column] *= -1.0
    return fixed


def _effective_components(value: int | str | float, *, source_rows: int, target_rows: int, feature_dim: int) -> int:
    max_components = max(1, min(feature_dim, max(1, source_rows - 1), max(1, target_rows - 1)))
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"all", "full", "inf", "infinity"}:
            return max_components
        requested = float(text)
    else:
        requested = float(value)
    if not np.isfinite(requested) or requested % 1.0 != 0.0 or requested < 1.0:
        raise ValueError("n_components must be a positive integer or 'all'.")
    return min(int(requested), max_components)


def _probabilities_or_none(model: BaseEstimator, features: np.ndarray) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        probs = np.asarray(model.predict_proba(features), dtype=float)
        return _normalize_probability_rows(probs)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features), dtype=float)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(np.clip(shifted, -50.0, 50.0))
        return _normalize_probability_rows(exp_scores)
    return None


def _predict_with_object_classes(model: BaseEstimator, features: np.ndarray) -> np.ndarray:
    predictions = model.predict(features)
    output = np.empty(len(predictions), dtype=object)
    for index, value in enumerate(list(predictions)):
        output[index] = value
    return output


def _classes_from_model(model: BaseEstimator, labels: np.ndarray) -> np.ndarray:
    values = getattr(model, "classes_", tuple(dict.fromkeys(labels.tolist())))
    vector = np.empty(len(values), dtype=object)
    for index, value in enumerate(list(values)):
        vector[index] = value
    return vector


def _normalize_probability_rows(probabilities: np.ndarray) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("probabilities must be a finite two-dimensional matrix.")
    matrix = np.maximum(matrix, 0.0)
    row_sums = np.sum(matrix, axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("probability rows must have positive mass.")
    return matrix / row_sums


def _safe_std(values: np.ndarray) -> np.ndarray:
    return np.maximum(np.std(values, axis=0, ddof=1 if values.shape[0] > 1 else 0), _MIN_SCALE)


def _format_vector(values: np.ndarray) -> str:
    return "|".join(f"{float(value):.12g}" for value in np.asarray(values, dtype=float).reshape(-1))


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    if matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one row and one feature column.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _object_vector(values: Sequence[Any] | np.ndarray, *, name: str) -> np.ndarray:
    if isinstance(values, np.ndarray) and values.dtype == object and values.ndim == 1:
        return values.reshape(-1)
    try:
        items = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a one-dimensional sequence.") from exc
    vector = np.empty(len(items), dtype=object)
    for index, value in enumerate(items):
        vector[index] = value
    return vector


def _positive_int(value: int | str, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(parsed)


def _positive_float(value: float | str, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be positive and finite.")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed
