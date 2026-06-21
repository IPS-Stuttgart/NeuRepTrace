"""Riemannian tangent-space transfer utilities for subject-adaptive M/EEG decoding.

The helpers in this module implement label-free Category-2 transfer pipelines:
source labels are used only by downstream classifiers and source-domain
regularizers, while held-out target recordings may be used to estimate covariance
recentering, tangent-space references, target graph structure, target distribution
matching, and pseudo-label refinement.  No target labels, class prototypes, or
target accuracies are required by these functions.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import eigh
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_RIEMANNIAN_EPSILON = 1.0e-9
DEFAULT_MEKT_REGULARIZATION = 1.0e-6
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


@dataclass(frozen=True, slots=True)
class MEKTTransferResult:
    """Projected feature matrices and provenance for full MEKT adaptation.

    ``source_features`` and ``target_features`` are the final MEKT projected
    features used by the downstream classifier.  ``source_tangent_features`` and
    ``target_tangent_features`` are the centroid-aligned tangent-space features
    before the MEKT generalized eigenproblem is solved.
    """

    source_features: np.ndarray
    target_features: np.ndarray
    source_tangent_features: np.ndarray
    target_tangent_features: np.ndarray
    source_projection: np.ndarray
    target_projection: np.ndarray
    initial_target_pseudo_labels: np.ndarray
    target_pseudo_labels: np.ndarray
    pseudo_label_history: tuple[np.ndarray, ...]
    generalized_eigenvalues: np.ndarray
    source_domains: np.ndarray
    selected_source_domains: np.ndarray
    domain_transferability: Mapping[Hashable, float]
    alpha: float
    beta: float
    rho: float
    n_neighbors: int
    n_iterations: int
    n_components: int
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


def mekt_tangent_transfer_features(
    source_covariances: Sequence[np.ndarray] | np.ndarray,
    target_covariances: Sequence[np.ndarray] | np.ndarray,
    *,
    source_domains: Sequence[Any] | np.ndarray | None = None,
    epsilon: float = DEFAULT_RIEMANNIAN_EPSILON,
) -> RiemannianTransferResult:
    """Return the centroid-aligned tangent features used by MEKT.

    MEKT maps each domain around its own centroid-alignment reference and then
    takes ``upper(log(P'))`` around the identity matrix.  This differs from
    ``riemannian_tangent_transfer_features(..., tangent_reference_scope=...)``,
    which may estimate a common tangent reference after alignment.
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
    aligned_target, target_reference = align_covariances_to_identity(target, epsilon=epsilon)

    source_features = _identity_tangent_features(aligned_source, epsilon=epsilon)
    target_features = _identity_tangent_features(aligned_target, epsilon=epsilon)
    return RiemannianTransferResult(
        source_features=source_features,
        target_features=target_features,
        source_reference=log_euclidean_mean(np.stack(source_references, axis=0), epsilon=epsilon),
        target_reference=target_reference,
        tangent_reference=np.eye(source.shape[1], dtype=float),
        source_domains=domain_ids,
    )


def mekt_transfer_features(
    source_covariances: Sequence[np.ndarray] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    target_covariances: Sequence[np.ndarray] | np.ndarray,
    *,
    source_domains: Sequence[Any] | np.ndarray | None = None,
    n_components: int = 10,
    n_iterations: int = 5,
    alpha: float = 1.0,
    beta: float = 0.1,
    rho: float = 1.0,
    n_neighbors: int = 5,
    graph_sigma: float = 1.0,
    source_domain_selection: int | None = None,
    estimator: Any | None = None,
    epsilon: float = DEFAULT_RIEMANNIAN_EPSILON,
    regularization: float = DEFAULT_MEKT_REGULARIZATION,
) -> MEKTTransferResult:
    """Run full unsupervised MEKT and return projected source/target features.

    This implements the MEKT pipeline from Zhang & Wu: domain-wise covariance
    centroid alignment, identity tangent-space mapping, joint-probability MMD
    using source labels and target pseudo-labels, source discriminability
    preservation, target graph-Laplacian locality preservation, parameter
    transfer/regularization, generalized eigen-decomposition, iterative target
    pseudo-label refinement, and optional domain transferability estimation (DTE)
    based source-domain selection.

    The target argument is deliberately limited to covariance matrices.  Target
    labels, target class prototypes, and target accuracy are never accepted by
    this Category-2 API.
    """

    labels = np.asarray(source_labels).reshape(-1)
    if labels.size == 0:
        raise ValueError("source_labels must contain at least one row.")
    alpha = _nonnegative_float(alpha, name="alpha")
    beta = _nonnegative_float(beta, name="beta")
    rho = _nonnegative_float(rho, name="rho")
    regularization = _positive_float(regularization, name="regularization")
    n_iterations = _positive_int(n_iterations, name="n_iterations")
    n_components = _positive_int(n_components, name="n_components")

    tangent = mekt_tangent_transfer_features(
        source_covariances,
        target_covariances,
        source_domains=source_domains,
        epsilon=epsilon,
    )
    if labels.shape[0] != tangent.source_features.shape[0]:
        raise ValueError("source_labels length must match source_covariances rows.")

    source_features = np.asarray(tangent.source_features, dtype=float)
    target_features = np.asarray(tangent.target_features, dtype=float)
    domain_ids = np.asarray(tangent.source_domains)
    transferability = estimate_domain_transferability(source_features, labels, target_features, source_domains=domain_ids)

    selected_domains = np.asarray(sorted(transferability, key=transferability.get, reverse=True), dtype=object)
    if source_domain_selection is not None:
        keep_count = _positive_int(source_domain_selection, name="source_domain_selection")
        unique_domains = np.unique(domain_ids)
        keep_count = min(keep_count, unique_domains.shape[0])
        selected_domains = np.asarray(sorted(transferability, key=transferability.get, reverse=True)[:keep_count], dtype=object)
        keep_mask = np.isin(domain_ids, selected_domains)
        source_features = source_features[keep_mask]
        labels = labels[keep_mask]
        domain_ids = domain_ids[keep_mask]

    n_components = min(n_components, source_features.shape[1])
    classifier = _default_estimator() if estimator is None else clone(estimator)
    classifier.fit(source_features, labels)
    initial_pseudo_labels = np.asarray(classifier.predict(target_features))
    pseudo_labels = initial_pseudo_labels.copy()
    pseudo_label_history: list[np.ndarray] = []
    eigenvalues = np.empty(0, dtype=float)
    source_projection = np.eye(source_features.shape[1], n_components, dtype=float)
    target_projection = np.eye(target_features.shape[1], n_components, dtype=float)
    projected_source = source_features @ source_projection
    projected_target = target_features @ target_projection

    for _ in range(n_iterations):
        source_projection, target_projection, eigenvalues = _solve_mekt_projection(
            source_features,
            labels,
            target_features,
            pseudo_labels,
            n_components=n_components,
            alpha=alpha,
            beta=beta,
            rho=rho,
            n_neighbors=n_neighbors,
            graph_sigma=graph_sigma,
            regularization=regularization,
        )
        projected_source = source_features @ source_projection
        projected_target = target_features @ target_projection
        classifier = _default_estimator() if estimator is None else clone(estimator)
        classifier.fit(projected_source, labels)
        pseudo_labels = np.asarray(classifier.predict(projected_target))
        pseudo_label_history.append(pseudo_labels.copy())

    return MEKTTransferResult(
        source_features=projected_source.astype(np.float32, copy=False),
        target_features=projected_target.astype(np.float32, copy=False),
        source_tangent_features=source_features.astype(np.float32, copy=False),
        target_tangent_features=target_features.astype(np.float32, copy=False),
        source_projection=source_projection.astype(np.float32, copy=False),
        target_projection=target_projection.astype(np.float32, copy=False),
        initial_target_pseudo_labels=initial_pseudo_labels,
        target_pseudo_labels=pseudo_labels,
        pseudo_label_history=tuple(pseudo_label_history),
        generalized_eigenvalues=eigenvalues.astype(np.float32, copy=False),
        source_domains=domain_ids,
        selected_source_domains=selected_domains,
        domain_transferability=transferability,
        alpha=alpha,
        beta=beta,
        rho=rho,
        n_neighbors=int(n_neighbors),
        n_iterations=int(n_iterations),
        n_components=int(n_components),
    )


def fit_predict_mekt(
    source_covariances: Sequence[np.ndarray] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    target_covariances: Sequence[np.ndarray] | np.ndarray,
    *,
    source_domains: Sequence[Any] | np.ndarray | None = None,
    estimator: Any | None = None,
    n_components: int = 10,
    n_iterations: int = 5,
    alpha: float = 1.0,
    beta: float = 0.1,
    rho: float = 1.0,
    n_neighbors: int = 5,
    graph_sigma: float = 1.0,
    source_domain_selection: int | None = None,
    epsilon: float = DEFAULT_RIEMANNIAN_EPSILON,
    regularization: float = DEFAULT_MEKT_REGULARIZATION,
) -> tuple[Any, MEKTTransferResult, np.ndarray]:
    """Fit a source-label classifier on MEKT features and predict target rows."""

    labels = np.asarray(source_labels).reshape(-1)
    transfer = mekt_transfer_features(
        source_covariances,
        labels,
        target_covariances,
        source_domains=source_domains,
        n_components=n_components,
        n_iterations=n_iterations,
        alpha=alpha,
        beta=beta,
        rho=rho,
        n_neighbors=n_neighbors,
        graph_sigma=graph_sigma,
        source_domain_selection=source_domain_selection,
        estimator=estimator,
        epsilon=epsilon,
        regularization=regularization,
    )
    if labels.shape[0] != np.asarray(source_covariances).shape[0]:
        raise ValueError("source_labels length must match source_covariances rows.")
    if source_domain_selection is not None:
        domain_ids = _domain_ids(labels.shape[0], source_domains)
        labels = labels[np.isin(domain_ids, transfer.selected_source_domains)]
    classifier = _default_estimator() if estimator is None else clone(estimator)
    classifier.fit(transfer.source_features, labels)
    predictions = np.asarray(classifier.predict(transfer.target_features))
    return classifier, transfer, predictions


def estimate_domain_transferability(
    source_features: Sequence[np.ndarray] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    target_features: Sequence[np.ndarray] | np.ndarray,
    *,
    source_domains: Sequence[Any] | np.ndarray | None = None,
    regularization: float = DEFAULT_MEKT_REGULARIZATION,
) -> dict[Hashable, float]:
    """Estimate MEKT domain transferability scores for source-domain selection.

    For each source domain, the numerator is its labeled between-class scatter
    magnitude and the denominator is the scatter between its class centroids and
    the unlabeled target centroid.  Higher scores indicate more discriminative
    source domains that are also closer to the target distribution.
    """

    source = _feature_matrix(source_features, name="source_features")
    target = _feature_matrix(target_features, name="target_features")
    labels = np.asarray(source_labels).reshape(-1)
    if labels.shape[0] != source.shape[0]:
        raise ValueError("source_labels length must match source_features rows.")
    if source.shape[1] != target.shape[1]:
        raise ValueError("source_features and target_features must have the same number of columns.")
    domain_ids = _domain_ids(source.shape[0], source_domains)
    target_mean = np.mean(target, axis=0)
    scores: dict[Hashable, float] = {}
    for domain in np.unique(domain_ids):
        mask = domain_ids == domain
        domain_features = source[mask]
        domain_labels = labels[mask]
        _, between = _source_scatter_matrices(domain_features, domain_labels)
        discriminability = float(np.sum(np.abs(between)))
        difference = _source_target_class_centroid_scatter(domain_features, domain_labels, target_mean)
        scores[domain.item() if hasattr(domain, "item") else domain] = discriminability / max(difference, regularization)
    return scores


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


def _identity_tangent_features(matrices: np.ndarray, *, epsilon: float) -> np.ndarray:
    return np.vstack([vectorize_symmetric(_matrix_log_spd(matrix, epsilon=epsilon), scale_off_diagonal=True) for matrix in matrices]).astype(np.float32, copy=False)


def _solve_mekt_projection(
    source_features: np.ndarray,
    source_labels: np.ndarray,
    target_features: np.ndarray,
    target_pseudo_labels: np.ndarray,
    *,
    n_components: int,
    alpha: float,
    beta: float,
    rho: float,
    n_neighbors: int,
    graph_sigma: float,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = _feature_matrix(source_features, name="source_features")
    target = _feature_matrix(target_features, name="target_features")
    labels = np.asarray(source_labels).reshape(-1)
    pseudo = np.asarray(target_pseudo_labels).reshape(-1)
    if labels.shape[0] != source.shape[0]:
        raise ValueError("source_labels length must match source_features rows.")
    if pseudo.shape[0] != target.shape[0]:
        raise ValueError("target_pseudo_labels length must match target_features rows.")
    if source.shape[1] != target.shape[1]:
        raise ValueError("source_features and target_features must have the same number of columns.")

    n_features = source.shape[1]
    n_target = target.shape[0]
    n_components = min(_positive_int(n_components, name="n_components"), n_features)

    within, between = _source_scatter_matrices(source, labels)
    laplacian = _target_graph_laplacian(target, n_neighbors=n_neighbors, sigma=graph_sigma)
    centering = _centering_matrix(n_target)
    x_source = source.T
    x_target = target.T

    p_block = _block_diag(within, np.zeros((n_features, n_features), dtype=float))
    locality_block = _block_diag(np.zeros((n_features, n_features), dtype=float), x_target @ laplacian @ x_target.T)
    identity = np.eye(n_features, dtype=float)
    transfer_block = np.block([[identity, -identity], [-identity, 2.0 * identity]])
    mmd_block = _joint_probability_mmd_block(x_source, labels, x_target, pseudo)
    left = _symmetrize(alpha * p_block + beta * locality_block + rho * transfer_block + mmd_block)

    target_constraint = x_target @ centering @ x_target.T
    right = _symmetrize(_block_diag(between, target_constraint))
    right += regularization * np.eye(right.shape[0], dtype=float)
    left += regularization * np.eye(left.shape[0], dtype=float)

    try:
        eigenvalues, eigenvectors = eigh(left, right, check_finite=False)
    except Exception:
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.pinv(right) @ left)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)

    order = np.argsort(eigenvalues)
    selected = order[:n_components]
    vectors = np.asarray(eigenvectors[:, selected], dtype=float)
    values = np.asarray(eigenvalues[selected], dtype=float)
    source_projection = vectors[:n_features]
    target_projection = vectors[n_features:]
    return source_projection, target_projection, values


def _joint_probability_mmd_block(x_source: np.ndarray, source_labels: np.ndarray, x_target: np.ndarray, target_pseudo_labels: np.ndarray) -> np.ndarray:
    n_source = x_source.shape[1]
    n_target = x_target.shape[1]
    classes = np.unique(source_labels)
    source_one_hot = _one_hot(source_labels, classes)
    target_one_hot = _one_hot(target_pseudo_labels, classes)
    ns = source_one_hot / max(1, n_source)
    nt = target_one_hot / max(1, n_target)
    return _symmetrize(
        np.block(
            [
                [x_source @ ns @ ns.T @ x_source.T, -x_source @ ns @ nt.T @ x_target.T],
                [-x_target @ nt @ ns.T @ x_source.T, x_target @ nt @ nt.T @ x_target.T],
            ]
        )
    )


def _source_scatter_matrices(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = _feature_matrix(features, name="features")
    label_array = np.asarray(labels).reshape(-1)
    if label_array.shape[0] != matrix.shape[0]:
        raise ValueError("labels length must match features rows.")
    n_features = matrix.shape[1]
    overall_mean = np.mean(matrix, axis=0)
    within = np.zeros((n_features, n_features), dtype=float)
    between = np.zeros((n_features, n_features), dtype=float)
    for label in np.unique(label_array):
        class_features = matrix[label_array == label]
        class_mean = np.mean(class_features, axis=0)
        centered = class_features - class_mean
        within += centered.T @ centered
        mean_difference = (class_mean - overall_mean).reshape(-1, 1)
        between += class_features.shape[0] * (mean_difference @ mean_difference.T)
    return _symmetrize(within), _symmetrize(between)


def _source_target_class_centroid_scatter(features: np.ndarray, labels: np.ndarray, target_mean: np.ndarray) -> float:
    matrix = _feature_matrix(features, name="features")
    label_array = np.asarray(labels).reshape(-1)
    target_center = np.asarray(target_mean, dtype=float).reshape(-1)
    if target_center.shape[0] != matrix.shape[1]:
        raise ValueError("target_mean length must match features columns.")
    scatter = np.zeros((matrix.shape[1], matrix.shape[1]), dtype=float)
    for label in np.unique(label_array):
        class_features = matrix[label_array == label]
        class_mean = np.mean(class_features, axis=0)
        mean_difference = (class_mean - target_center).reshape(-1, 1)
        scatter += class_features.shape[0] * (mean_difference @ mean_difference.T)
    return float(np.sum(np.abs(scatter)))


def _target_graph_laplacian(features: np.ndarray, *, n_neighbors: int, sigma: float) -> np.ndarray:
    target = _feature_matrix(features, name="target_features")
    n_rows = target.shape[0]
    if n_rows <= 1:
        return np.zeros((n_rows, n_rows), dtype=float)
    sigma = _positive_float(sigma, name="graph_sigma")
    neighbor_count = min(_positive_int(n_neighbors, name="n_neighbors"), n_rows - 1)
    difference = target[:, None, :] - target[None, :, :]
    distances = np.sum(difference * difference, axis=2)
    similarity = np.zeros((n_rows, n_rows), dtype=float)
    for row in range(n_rows):
        neighbors = np.argsort(distances[row])[1 : neighbor_count + 1]
        weights = np.exp(-distances[row, neighbors] / (2.0 * sigma * sigma))
        similarity[row, neighbors] = weights
    similarity = np.maximum(similarity, similarity.T)
    degree = similarity.sum(axis=1)
    laplacian = np.eye(n_rows, dtype=float)
    nonzero = degree > np.finfo(float).eps
    inv_sqrt_degree = np.zeros_like(degree)
    inv_sqrt_degree[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    laplacian -= (inv_sqrt_degree[:, None] * similarity) * inv_sqrt_degree[None, :]
    return _symmetrize(laplacian)


def _one_hot(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    label_array = np.asarray(labels).reshape(-1)
    class_array = np.asarray(classes)
    encoded = np.zeros((label_array.shape[0], class_array.shape[0]), dtype=float)
    for index, label in enumerate(label_array):
        matches = np.flatnonzero(class_array == label)
        if matches.size:
            encoded[index, matches[0]] = 1.0
    return encoded


def _centering_matrix(n_rows: int) -> np.ndarray:
    if n_rows < 1:
        raise ValueError("n_rows must be positive.")
    return np.eye(n_rows, dtype=float) - np.full((n_rows, n_rows), 1.0 / n_rows, dtype=float)


def _block_diag(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    top = np.concatenate([a, np.zeros((a.shape[0], b.shape[1]), dtype=float)], axis=1)
    bottom = np.concatenate([np.zeros((b.shape[0], a.shape[1]), dtype=float), b], axis=1)
    return np.concatenate([top, bottom], axis=0)


def _feature_matrix(features: Sequence[np.ndarray] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must have shape n_rows x n_features.")
    if matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one row and one feature.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


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


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    number = int(value)
    if number != float(value) or number <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return number


def _positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive finite value.")
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite value.")
    return number


def _nonnegative_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a nonnegative finite value.")
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a nonnegative finite value.")
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


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
