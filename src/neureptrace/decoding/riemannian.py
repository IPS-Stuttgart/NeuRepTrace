"""Riemannian tangent-space transfer utilities for subject-adaptive M/EEG decoding.

The helpers in this module implement a label-free Category-2 transfer pipeline:
source labels are used only by the downstream classifier, while held-out target
recordings may be used to estimate covariance recentering and tangent-space
reference matrices.  No target labels, class prototypes, or target accuracies are
required by these functions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_RIEMANNIAN_EPSILON = 1.0e-9
TANGENT_REFERENCE_SCOPES = ("source", "source_target")


@dataclass(frozen=True, slots=True)
class RiemannianTransferResult:
    """Feature matrices and provenance for one Category-2 transfer fit."""

    source_features: np.ndarray
    target_features: np.ndarray
    source_reference: np.ndarray
    target_reference: np.ndarray
    tangent_reference: np.ndarray
    source_domains: np.ndarray
    protocol_category: int = 2
    uses_target_labels: bool = False
    uses_target_features: bool = True


def normalize_tangent_reference_scope(value: Any = "source") -> str:
    """Normalize the tangent-reference scope used after domain alignment."""

    normalized = "source" if value is None else str(value).strip().lower().replace("-", "_")
    aliases = {
        "train": "source",
        "training": "source",
        "source_only": "source",
        "source+target": "source_target",
        "source_and_target": "source_target",
        "train_target": "source_target",
        "all_unlabeled": "source_target",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in TANGENT_REFERENCE_SCOPES:
        raise ValueError(f"tangent_reference_scope must be one of {TANGENT_REFERENCE_SCOPES}, got {value!r}.")
    return normalized


def ensure_spd_matrices(matrices: Sequence[np.ndarray] | np.ndarray, *, epsilon: float = DEFAULT_RIEMANNIAN_EPSILON, name: str = "covariances") -> np.ndarray:
    """Return symmetrized positive-definite covariance matrices.

    Eigenvalues are floored relative to each matrix trace.  This keeps tangent
    maps stable for small synthetic tests and short-window covariance estimates
    without changing the public dependency footprint by requiring pyRiemann.
    """

    array = np.asarray(matrices, dtype=float)
    if array.ndim != 3 or array.shape[1] != array.shape[2]:
        raise ValueError(f"{name} must have shape n_matrices x n_channels x n_channels.")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one square matrix.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    epsilon = _positive_float(epsilon, name="epsilon")
    return np.stack([_nearest_spd(matrix, epsilon=epsilon) for matrix in array], axis=0)


def log_euclidean_mean(matrices: Sequence[np.ndarray] | np.ndarray, *, epsilon: float = DEFAULT_RIEMANNIAN_EPSILON) -> np.ndarray:
    """Return the log-Euclidean mean of SPD matrices."""

    spd = ensure_spd_matrices(matrices, epsilon=epsilon)
    return _matrix_exp_spd(np.mean([_matrix_log_spd(matrix, epsilon=epsilon) for matrix in spd], axis=0), epsilon=epsilon)


def align_covariances_to_identity(
    matrices: Sequence[np.ndarray] | np.ndarray,
    *,
    reference: np.ndarray | None = None,
    epsilon: float = DEFAULT_RIEMANNIAN_EPSILON,
) -> tuple[np.ndarray, np.ndarray]:
    """Recenter SPD matrices by whitening their unlabeled domain reference.

    Returns ``(aligned_matrices, reference)``.  When ``reference`` is omitted,
    the log-Euclidean mean of ``matrices`` is used.  This is the standard
    label-free Euclidean/Riemannian alignment step used before tangent-space
    transfer classifiers.
    """

    spd = ensure_spd_matrices(matrices, epsilon=epsilon)
    domain_reference = log_euclidean_mean(spd, epsilon=epsilon) if reference is None else _nearest_spd(reference, epsilon=epsilon)
    whitener = _matrix_inv_sqrt_spd(domain_reference, epsilon=epsilon)
    aligned = np.stack([_nearest_spd(whitener @ matrix @ whitener.T, epsilon=epsilon) for matrix in spd], axis=0)
    return aligned, domain_reference


def tangent_space_features(
    matrices: Sequence[np.ndarray] | np.ndarray,
    *,
    reference: np.ndarray | None = None,
    epsilon: float = DEFAULT_RIEMANNIAN_EPSILON,
) -> tuple[np.ndarray, np.ndarray]:
    """Map SPD matrices to symmetric tangent-space vectors."""

    spd = ensure_spd_matrices(matrices, epsilon=epsilon)
    tangent_reference = log_euclidean_mean(spd, epsilon=epsilon) if reference is None else _nearest_spd(reference, epsilon=epsilon)
    whitener = _matrix_inv_sqrt_spd(tangent_reference, epsilon=epsilon)
    vectors = []
    for matrix in spd:
        tangent_matrix = _matrix_log_spd(whitener @ matrix @ whitener.T, epsilon=epsilon)
        vectors.append(vectorize_symmetric(tangent_matrix, scale_off_diagonal=True))
    return np.vstack(vectors).astype(np.float32, copy=False), tangent_reference


def riemannian_tangent_transfer_features(
    source_covariances: Sequence[np.ndarray] | np.ndarray,
    target_covariances: Sequence[np.ndarray] | np.ndarray,
    *,
    source_domains: Sequence[Any] | np.ndarray | None = None,
    tangent_reference_scope: str = "source",
    epsilon: float = DEFAULT_RIEMANNIAN_EPSILON,
) -> RiemannianTransferResult:
    """Return Category-2 source/target tangent features after domain recentering.

    Parameters
    ----------
    source_covariances, target_covariances:
        Trial-level SPD covariance matrices.  The target matrices are treated as
        unlabeled calibration/adaptation data; target labels are intentionally not
        accepted by this API.
    source_domains:
        Optional domain ids for source trials, usually source-subject ids.  When
        provided, each source domain is recentered by its own unlabeled covariance
        reference before all source trials are pooled.  When omitted, all source
        covariances share one source reference.
    tangent_reference_scope:
        ``"source"`` fits the tangent reference after alignment from source
        covariances only.  ``"source_target"`` additionally uses unlabeled target
        covariances and is therefore a stronger transductive Category-2 setting.
    """

    source = ensure_spd_matrices(source_covariances, epsilon=epsilon, name="source_covariances")
    target = ensure_spd_matrices(target_covariances, epsilon=epsilon, name="target_covariances")
    if source.shape[1:] != target.shape[1:]:
        raise ValueError("source_covariances and target_covariances must use the same channel dimensions.")

    domain_ids = _domain_ids(source.shape[0], source_domains)
    aligned_source = np.empty_like(source)
    source_references = []
    for domain in np.unique(domain_ids):
        mask = domain_ids == domain
        aligned_domain, domain_reference = align_covariances_to_identity(source[mask], epsilon=epsilon)
        aligned_source[mask] = aligned_domain
        source_references.append(domain_reference)
    source_reference = log_euclidean_mean(np.stack(source_references, axis=0), epsilon=epsilon)
    aligned_target, target_reference = align_covariances_to_identity(target, epsilon=epsilon)

    reference_scope = normalize_tangent_reference_scope(tangent_reference_scope)
    reference_input = aligned_source if reference_scope == "source" else np.concatenate([aligned_source, aligned_target], axis=0)
    tangent_reference = log_euclidean_mean(reference_input, epsilon=epsilon)
    source_features, _ = tangent_space_features(aligned_source, reference=tangent_reference, epsilon=epsilon)
    target_features, _ = tangent_space_features(aligned_target, reference=tangent_reference, epsilon=epsilon)
    return RiemannianTransferResult(
        source_features=source_features,
        target_features=target_features,
        source_reference=source_reference,
        target_reference=target_reference,
        tangent_reference=tangent_reference,
        source_domains=domain_ids,
    )


def fit_predict_riemannian_transfer(
    source_covariances: Sequence[np.ndarray] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    target_covariances: Sequence[np.ndarray] | np.ndarray,
    *,
    source_domains: Sequence[Any] | np.ndarray | None = None,
    estimator: Any | None = None,
    tangent_reference_scope: str = "source",
    epsilon: float = DEFAULT_RIEMANNIAN_EPSILON,
) -> tuple[Any, RiemannianTransferResult, np.ndarray]:
    """Fit a source-label classifier in Category-2 tangent space and predict target rows."""

    labels = np.asarray(source_labels).reshape(-1)
    transfer = riemannian_tangent_transfer_features(
        source_covariances,
        target_covariances,
        source_domains=source_domains,
        tangent_reference_scope=tangent_reference_scope,
        epsilon=epsilon,
    )
    if labels.shape[0] != transfer.source_features.shape[0]:
        raise ValueError("source_labels length must match source_covariances rows.")
    classifier = _default_estimator() if estimator is None else clone(estimator)
    classifier.fit(transfer.source_features, labels)
    predictions = np.asarray(classifier.predict(transfer.target_features))
    return classifier, transfer, predictions


def vectorize_symmetric(matrix: np.ndarray, *, scale_off_diagonal: bool = True) -> np.ndarray:
    """Vectorize the upper triangle of a symmetric matrix."""

    symmetric = np.asarray(matrix, dtype=float)
    if symmetric.ndim != 2 or symmetric.shape[0] != symmetric.shape[1]:
        raise ValueError("matrix must be square.")
    rows, cols = np.triu_indices(symmetric.shape[0])
    values = symmetric[rows, cols].astype(float, copy=True)
    if scale_off_diagonal:
        values[rows != cols] *= np.sqrt(2.0)
    return values


def _default_estimator():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=13),
    )


def _domain_ids(n_rows: int, source_domains: Sequence[Any] | np.ndarray | None) -> np.ndarray:
    if source_domains is None:
        return np.zeros(n_rows, dtype=int)
    domains = np.asarray(source_domains)
    if domains.shape[0] != n_rows:
        raise ValueError("source_domains length must match source_covariances rows.")
    return domains


def _positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive finite value.")
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite value.")
    return number


def _nearest_spd(matrix: np.ndarray, *, epsilon: float) -> np.ndarray:
    symmetric = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    if symmetric.ndim != 2 or symmetric.shape[0] != symmetric.shape[1]:
        raise ValueError("SPD matrices must be square.")
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = float(np.trace(symmetric) / max(1, symmetric.shape[0]))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    floored = np.maximum(eigenvalues, epsilon * scale)
    repaired = (eigenvectors * floored[None, :]) @ eigenvectors.T
    return 0.5 * (repaired + repaired.T)


def _matrix_log_spd(matrix: np.ndarray, *, epsilon: float) -> np.ndarray:
    spd = _nearest_spd(matrix, epsilon=epsilon)
    eigenvalues, eigenvectors = np.linalg.eigh(spd)
    log_values = np.log(np.maximum(eigenvalues, epsilon))
    logged = (eigenvectors * log_values[None, :]) @ eigenvectors.T
    return 0.5 * (logged + logged.T)


def _matrix_exp_spd(matrix: np.ndarray, *, epsilon: float) -> np.ndarray:
    symmetric = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    exp_values = np.exp(eigenvalues)
    return _nearest_spd((eigenvectors * exp_values[None, :]) @ eigenvectors.T, epsilon=epsilon)


def _matrix_inv_sqrt_spd(matrix: np.ndarray, *, epsilon: float) -> np.ndarray:
    spd = _nearest_spd(matrix, epsilon=epsilon)
    eigenvalues, eigenvectors = np.linalg.eigh(spd)
    inv_sqrt_values = 1.0 / np.sqrt(np.maximum(eigenvalues, epsilon))
    inv_sqrt = (eigenvectors * inv_sqrt_values[None, :]) @ eigenvectors.T
    return 0.5 * (inv_sqrt + inv_sqrt.T)
