"""Manifold embedded knowledge transfer (MEKT) for cross-subject M/EEG decoding.

The implementation follows the linear MEKT protocol: covariance-centroid
alignment, identity tangent-space mapping, optional domain transferability
estimation (DTE), source/target projection learning by generalized
eigendecomposition, and iterative pseudo-label refinement.  It is a Category-2
protocol: target recordings may be used as unlabeled calibration/adaptation data,
but target labels are never accepted by this API.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import eigh
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from neureptrace.decoding.riemannian import (
    DEFAULT_RIEMANNIAN_EPSILON,
    align_covariances_to_identity,
    ensure_spd_matrices,
    log_euclidean_mean,
    tangent_space_features,
)

DEFAULT_MEKT_ALPHA = 0.1
DEFAULT_MEKT_BETA = 1.0
DEFAULT_MEKT_RHO = 20.0
DEFAULT_MEKT_ITERATIONS = 5
DEFAULT_MEKT_COMPONENTS = 16
DEFAULT_MEKT_NEIGHBORS = 5
DEFAULT_MEKT_GRAPH_SIGMA = 1.0


@dataclass(frozen=True, slots=True)
class MektCentroidFeatures:
    """MEKT covariance-centroid-aligned tangent features."""

    source_features: np.ndarray
    target_features: np.ndarray
    source_reference: np.ndarray
    target_reference: np.ndarray
    source_domains: np.ndarray
    protocol_category: int = 2
    uses_target_labels: bool = False
    uses_target_features: bool = True


@dataclass(frozen=True, slots=True)
class MektTransferResult:
    """Projected MEKT features and provenance for one Category-2 transfer fit."""

    source_features: np.ndarray
    target_features: np.ndarray
    source_embedding: np.ndarray
    target_embedding: np.ndarray
    source_projection: np.ndarray
    target_projection: np.ndarray
    source_reference: np.ndarray
    target_reference: np.ndarray
    source_domains: np.ndarray
    selected_source_domains: np.ndarray
    transferability_scores: Mapping[Any, float]
    classes: np.ndarray
    pseudo_labels: np.ndarray
    pseudo_label_history: tuple[np.ndarray, ...]
    n_components: int
    n_iterations: int
    alpha: float
    beta: float
    rho: float
    n_neighbors: int
    graph_sigma: float
    protocol_category: int = 2
    uses_target_labels: bool = False
    uses_target_features: bool = True


def centroid_aligned_tangent_features(
    source_covariances: Sequence[np.ndarray] | np.ndarray,
    target_covariances: Sequence[np.ndarray] | np.ndarray,
    *,
    source_domains: Sequence[Any] | np.ndarray | None = None,
    epsilon: float = DEFAULT_RIEMANNIAN_EPSILON,
) -> MektCentroidFeatures:
    """Return MEKT Step 1-2 features: CA on each domain, then ``log(P')``.

    Target covariances are used only to estimate the unlabeled target centroid.
    Source domains, when provided, are recentered separately before pooling.
    """

    source = ensure_spd_matrices(source_covariances, epsilon=epsilon, name="source_covariances")
    target = ensure_spd_matrices(target_covariances, epsilon=epsilon, name="target_covariances")
    if source.shape[1:] != target.shape[1:]:
        raise ValueError("source_covariances and target_covariances must use the same channel dimensions.")

    domain_ids = _domain_ids(source.shape[0], source_domains, name="source_domains")
    aligned_source = np.empty_like(source)
    source_references = []
    for domain in np.unique(domain_ids):
        mask = domain_ids == domain
        aligned_domain, reference = align_covariances_to_identity(source[mask], epsilon=epsilon)
        aligned_source[mask] = aligned_domain
        source_references.append(reference)
    aligned_target, target_reference = align_covariances_to_identity(target, epsilon=epsilon)
    source_reference = log_euclidean_mean(np.stack(source_references, axis=0), epsilon=epsilon)
    identity = np.eye(source.shape[1], dtype=float)
    source_features, _ = tangent_space_features(aligned_source, reference=identity, epsilon=epsilon)
    target_features, _ = tangent_space_features(aligned_target, reference=identity, epsilon=epsilon)
    return MektCentroidFeatures(
        source_features=source_features,
        target_features=target_features,
        source_reference=source_reference,
        target_reference=target_reference,
        source_domains=domain_ids,
    )


def mekt_transfer_features(
    source_covariances: Sequence[np.ndarray] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    target_covariances: Sequence[np.ndarray] | np.ndarray,
    *,
    source_domains: Sequence[Any] | np.ndarray | None = None,
    estimator: Any | None = None,
    n_components: int | None = None,
    n_iterations: int = DEFAULT_MEKT_ITERATIONS,
    alpha: float = DEFAULT_MEKT_ALPHA,
    beta: float = DEFAULT_MEKT_BETA,
    rho: float = DEFAULT_MEKT_RHO,
    n_neighbors: int = DEFAULT_MEKT_NEIGHBORS,
    graph_sigma: float = DEFAULT_MEKT_GRAPH_SIGMA,
    dte_top_k: int | None = None,
    initial_pseudo_labels: Sequence[Any] | np.ndarray | None = None,
    epsilon: float = DEFAULT_RIEMANNIAN_EPSILON,
) -> MektTransferResult:
    """Fit full linear MEKT using source labels plus unlabeled target covariances."""

    source = ensure_spd_matrices(source_covariances, epsilon=epsilon, name="source_covariances")
    target = ensure_spd_matrices(target_covariances, epsilon=epsilon, name="target_covariances")
    labels = np.asarray(source_labels).reshape(-1)
    if labels.shape[0] != source.shape[0]:
        raise ValueError("source_labels length must match source_covariances rows.")
    classes = np.unique(labels)
    if classes.shape[0] < 2:
        raise ValueError("MEKT requires at least two source classes.")

    n_iterations = _positive_int(n_iterations, name="n_iterations")
    alpha = _nonnegative_float(alpha, name="alpha")
    beta = _nonnegative_float(beta, name="beta")
    rho = _positive_float(rho, name="rho")
    n_neighbors = _positive_int(n_neighbors, name="n_neighbors")
    graph_sigma = _positive_float(graph_sigma, name="graph_sigma")
    epsilon = _positive_float(epsilon, name="epsilon")

    domain_ids = _domain_ids(source.shape[0], source_domains, name="source_domains")
    selected_domains = np.unique(domain_ids)
    transferability_scores: dict[Any, float] = {}

    if dte_top_k is not None:
        top_k = _positive_int(dte_top_k, name="dte_top_k")
        preliminary = centroid_aligned_tangent_features(source, target, source_domains=domain_ids, epsilon=epsilon)
        transferability_scores = domain_transferability_scores(
            preliminary.source_features,
            labels,
            preliminary.target_features,
            domain_ids,
            epsilon=epsilon,
        )
        selected_domains = _select_top_domains(transferability_scores, top_k)
        keep_mask = np.isin(domain_ids, selected_domains)
        if np.unique(labels[keep_mask]).shape[0] < 2:
            raise ValueError("DTE source-domain selection left fewer than two source classes.")
        source = source[keep_mask]
        labels = labels[keep_mask]
        domain_ids = domain_ids[keep_mask]

    base = centroid_aligned_tangent_features(source, target, source_domains=domain_ids, epsilon=epsilon)
    n_components_resolved = _normalize_components(n_components, base.source_features.shape[1])
    classifier_template = _default_estimator() if estimator is None else estimator
    pseudo = _initial_pseudo_labels(
        base.source_features,
        labels,
        base.target_features,
        classes=classes,
        estimator=classifier_template,
        initial_pseudo_labels=initial_pseudo_labels,
    )

    history: list[np.ndarray] = [pseudo.copy()]
    source_projection = np.zeros((base.source_features.shape[1], n_components_resolved), dtype=float)
    target_projection = np.zeros_like(source_projection)
    source_embedding = base.source_features[:, :n_components_resolved].copy()
    target_embedding = base.target_features[:, :n_components_resolved].copy()

    for _ in range(n_iterations):
        source_projection, target_projection = _solve_mekt_projection(
            base.source_features,
            labels,
            base.target_features,
            pseudo,
            classes=classes,
            n_components=n_components_resolved,
            alpha=alpha,
            beta=beta,
            rho=rho,
            n_neighbors=n_neighbors,
            graph_sigma=graph_sigma,
            epsilon=epsilon,
        )
        source_embedding = base.source_features @ source_projection
        target_embedding = base.target_features @ target_projection
        classifier = clone(classifier_template)
        classifier.fit(source_embedding, labels)
        new_pseudo = np.asarray(classifier.predict(target_embedding))
        history.append(new_pseudo.copy())
        if np.array_equal(new_pseudo, pseudo):
            pseudo = new_pseudo
            break
        pseudo = new_pseudo

    return MektTransferResult(
        source_features=base.source_features,
        target_features=base.target_features,
        source_embedding=source_embedding.astype(np.float32, copy=False),
        target_embedding=target_embedding.astype(np.float32, copy=False),
        source_projection=source_projection.astype(np.float32, copy=False),
        target_projection=target_projection.astype(np.float32, copy=False),
        source_reference=base.source_reference,
        target_reference=base.target_reference,
        source_domains=domain_ids,
        selected_source_domains=np.asarray(selected_domains),
        transferability_scores=transferability_scores,
        classes=classes,
        pseudo_labels=pseudo,
        pseudo_label_history=tuple(history),
        n_components=n_components_resolved,
        n_iterations=len(history) - 1,
        alpha=alpha,
        beta=beta,
        rho=rho,
        n_neighbors=n_neighbors,
        graph_sigma=graph_sigma,
    )


def fit_predict_mekt_transfer(
    source_covariances: Sequence[np.ndarray] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    target_covariances: Sequence[np.ndarray] | np.ndarray,
    *,
    source_domains: Sequence[Any] | np.ndarray | None = None,
    estimator: Any | None = None,
    n_components: int | None = None,
    n_iterations: int = DEFAULT_MEKT_ITERATIONS,
    alpha: float = DEFAULT_MEKT_ALPHA,
    beta: float = DEFAULT_MEKT_BETA,
    rho: float = DEFAULT_MEKT_RHO,
    n_neighbors: int = DEFAULT_MEKT_NEIGHBORS,
    graph_sigma: float = DEFAULT_MEKT_GRAPH_SIGMA,
    dte_top_k: int | None = None,
    epsilon: float = DEFAULT_RIEMANNIAN_EPSILON,
) -> tuple[Any, MektTransferResult, np.ndarray]:
    """Fit a source-supervised classifier in MEKT space and predict target rows."""

    classifier_template = _default_estimator() if estimator is None else estimator
    transfer = mekt_transfer_features(
        source_covariances,
        source_labels,
        target_covariances,
        source_domains=source_domains,
        estimator=classifier_template,
        n_components=n_components,
        n_iterations=n_iterations,
        alpha=alpha,
        beta=beta,
        rho=rho,
        n_neighbors=n_neighbors,
        graph_sigma=graph_sigma,
        dte_top_k=dte_top_k,
        epsilon=epsilon,
    )
    labels = np.asarray(source_labels).reshape(-1)
    if transfer.source_domains.shape[0] != labels.shape[0]:
        domains = _domain_ids(labels.shape[0], source_domains, name="source_domains")
        labels = labels[np.isin(domains, transfer.selected_source_domains)]
    classifier = clone(classifier_template)
    classifier.fit(transfer.source_embedding, labels)
    predictions = np.asarray(classifier.predict(transfer.target_embedding))
    return classifier, transfer, predictions


def domain_transferability_scores(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    target_features: Sequence[Sequence[float]] | np.ndarray,
    source_domains: Sequence[Any] | np.ndarray,
    *,
    epsilon: float = DEFAULT_RIEMANNIAN_EPSILON,
) -> dict[Any, float]:
    """Return MEKT DTE scores ``DIS(S_i) / DIF(S_i, T)`` for source domains."""

    source = _as_2d_float(source_features, name="source_features")
    target = _as_2d_float(target_features, name="target_features")
    labels = np.asarray(source_labels).reshape(-1)
    domains = _domain_ids(source.shape[0], source_domains, name="source_domains")
    if labels.shape[0] != source.shape[0]:
        raise ValueError("source_labels length must match source_features rows.")
    if source.shape[1] != target.shape[1]:
        raise ValueError("source_features and target_features must have the same number of columns.")
    target_mean = target.mean(axis=0)
    scores: dict[Any, float] = {}
    for domain in np.unique(domains):
        mask = domains == domain
        dis = float(np.linalg.norm(_between_class_scatter(source[mask], labels[mask]), ord=1))
        dif = float(np.linalg.norm(_two_domain_between_scatter(source[mask], target_mean, target.shape[0]), ord=1))
        key = domain.item() if isinstance(domain, np.generic) else domain
        scores[key] = dis / (dif + epsilon)
    return scores


def _initial_pseudo_labels(
    source_features: np.ndarray,
    source_labels: np.ndarray,
    target_features: np.ndarray,
    *,
    classes: np.ndarray,
    estimator: Any,
    initial_pseudo_labels: Sequence[Any] | np.ndarray | None,
) -> np.ndarray:
    if initial_pseudo_labels is None:
        classifier = clone(estimator)
        classifier.fit(source_features, source_labels)
        return np.asarray(classifier.predict(target_features))
    pseudo = np.asarray(initial_pseudo_labels).reshape(-1)
    if pseudo.shape[0] != target_features.shape[0]:
        raise ValueError("initial_pseudo_labels length must match target_covariances rows.")
    if not np.all(np.isin(pseudo, classes)):
        raise ValueError("initial_pseudo_labels must contain only source classes.")
    return pseudo


def _solve_mekt_projection(
    source_features: np.ndarray,
    source_labels: np.ndarray,
    target_features: np.ndarray,
    target_pseudo_labels: np.ndarray,
    *,
    classes: np.ndarray,
    n_components: int,
    alpha: float,
    beta: float,
    rho: float,
    n_neighbors: int,
    graph_sigma: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    source = _as_2d_float(source_features, name="source_features")
    target = _as_2d_float(target_features, name="target_features")
    labels = np.asarray(source_labels).reshape(-1)
    pseudo = np.asarray(target_pseudo_labels).reshape(-1)
    if labels.shape[0] != source.shape[0]:
        raise ValueError("source_labels length must match source_features rows.")
    if pseudo.shape[0] != target.shape[0]:
        raise ValueError("target_pseudo_labels length must match target_features rows.")
    if source.shape[1] != target.shape[1]:
        raise ValueError("source_features and target_features must have the same number of columns.")

    n_features = source.shape[1]
    zeros = np.zeros((n_features, n_features), dtype=float)
    identity = np.eye(n_features, dtype=float)

    sw = _within_class_scatter(source, labels)
    sb = _between_class_scatter(source, labels)
    laplacian = _normalized_knn_laplacian(target, n_neighbors=n_neighbors, sigma=graph_sigma)
    target_centering = _centering_matrix(target.shape[0])
    target_locality = target.T @ laplacian @ target
    target_variance = target.T @ target_centering @ target

    ns = _one_hot(labels, classes) / float(source.shape[0])
    nt = _one_hot(pseudo, classes) / float(target.shape[0])
    r_top_left = source.T @ ns @ ns.T @ source
    r_top_right = -(source.T @ ns @ nt.T @ target)
    r_matrix = np.block(
        [
            [r_top_left, r_top_right],
            [r_top_right.T, target.T @ nt @ nt.T @ target],
        ]
    )

    p_matrix = np.block([[sw, zeros], [zeros, zeros]])
    l_matrix = np.block([[zeros, zeros], [zeros, target_locality]])
    u_matrix = np.block([[identity, -identity], [-identity, 2.0 * identity]])
    v_matrix = np.block([[sb, zeros], [zeros, target_variance]])
    left = alpha * p_matrix + beta * l_matrix + rho * u_matrix + r_matrix
    w = _generalized_eigh_smallest(left, v_matrix, n_components=n_components, epsilon=epsilon)
    return w[:n_features, :], w[n_features:, :]


def _within_class_scatter(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    x = _as_2d_float(features, name="features")
    y = np.asarray(labels).reshape(-1)
    scatter = np.zeros((x.shape[1], x.shape[1]), dtype=float)
    for label in np.unique(y):
        class_features = x[y == label]
        if class_features.shape[0] <= 1:
            continue
        centered = class_features - class_features.mean(axis=0, keepdims=True)
        scatter += centered.T @ centered
    return _symmetrize(scatter)


def _between_class_scatter(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    x = _as_2d_float(features, name="features")
    y = np.asarray(labels).reshape(-1)
    overall = x.mean(axis=0)
    scatter = np.zeros((x.shape[1], x.shape[1]), dtype=float)
    for label in np.unique(y):
        class_features = x[y == label]
        diff = (class_features.mean(axis=0) - overall).reshape(-1, 1)
        scatter += float(class_features.shape[0]) * (diff @ diff.T)
    return _symmetrize(scatter)


def _two_domain_between_scatter(source_features: np.ndarray, target_mean: np.ndarray, n_target: int) -> np.ndarray:
    source = _as_2d_float(source_features, name="source_features")
    target_mean = np.asarray(target_mean, dtype=float).reshape(-1)
    n_source = source.shape[0]
    source_mean = source.mean(axis=0)
    combined = (float(n_source) * source_mean + float(n_target) * target_mean) / float(n_source + n_target)
    source_diff = (source_mean - combined).reshape(-1, 1)
    target_diff = (target_mean - combined).reshape(-1, 1)
    return _symmetrize(float(n_source) * (source_diff @ source_diff.T) + float(n_target) * (target_diff @ target_diff.T))


def _normalized_knn_laplacian(features: np.ndarray, *, n_neighbors: int, sigma: float) -> np.ndarray:
    x = _as_2d_float(features, name="target_features")
    n_rows = x.shape[0]
    if n_rows == 1:
        return np.zeros((1, 1), dtype=float)
    k = min(max(1, int(n_neighbors)), n_rows - 1)
    distances = _squared_euclidean_distances(x)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    similarity = np.zeros((n_rows, n_rows), dtype=float)
    scale = 2.0 * float(sigma) ** 2
    for row in range(n_rows):
        for col in neighbors[row]:
            weight = float(np.exp(-distances[row, col] / scale))
            similarity[row, col] = max(similarity[row, col], weight)
            similarity[col, row] = max(similarity[col, row], weight)
    degree = similarity.sum(axis=1)
    laplacian = np.eye(n_rows, dtype=float)
    positive = degree > 0.0
    if np.any(positive):
        inv_sqrt_degree = np.zeros_like(degree)
        inv_sqrt_degree[positive] = 1.0 / np.sqrt(degree[positive])
        laplacian -= (inv_sqrt_degree[:, None] * similarity) * inv_sqrt_degree[None, :]
    return _symmetrize(laplacian)


def _generalized_eigh_smallest(left: np.ndarray, right: np.ndarray, *, n_components: int, epsilon: float) -> np.ndarray:
    a = _symmetrize(left)
    b = _symmetrize(right)
    n_rows = a.shape[0]
    scale = float(np.trace(b) / max(1, n_rows))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    identity = np.eye(n_rows, dtype=float)
    last_error: Exception | None = None
    for multiplier in (1.0, 10.0, 100.0, 1000.0, 10000.0):
        try:
            values, vectors = eigh(a + epsilon * multiplier * identity, b + epsilon * multiplier * scale * identity, check_finite=True)
            return _normalize_columns(np.real(vectors[:, np.argsort(np.real(values))[:n_components]]))
        except Exception as exc:  # pragma: no cover
            last_error = exc
    try:
        operator = np.linalg.pinv(b + epsilon * scale * identity) @ a
        values, vectors = np.linalg.eig(operator)
        return _normalize_columns(np.real(vectors[:, np.argsort(np.real(values))[:n_components]]))
    except Exception as exc:  # pragma: no cover
        raise np.linalg.LinAlgError(f"MEKT generalized eigenproblem failed: {last_error!r}; fallback failed: {exc!r}") from exc


def _one_hot(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    encoded = np.zeros((labels.shape[0], classes.shape[0]), dtype=float)
    for class_index, label in enumerate(classes):
        encoded[labels == label, class_index] = 1.0
    if np.any(encoded.sum(axis=1) == 0.0):
        raise ValueError("labels contain values that are not present in classes.")
    return encoded


def _normalize_components(value: int | None, n_features: int) -> int:
    if value is None:
        return min(DEFAULT_MEKT_COMPONENTS, max(1, int(n_features)))
    return min(_positive_int(value, name="n_components"), max(1, int(n_features)))


def _select_top_domains(scores: Mapping[Any, float], top_k: int) -> np.ndarray:
    ordered = sorted(scores, key=lambda key: (-float(scores[key]), str(key)))
    return np.asarray(ordered[: min(int(top_k), len(ordered))])


def _as_2d_float(features: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(features, dtype=float)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} must have shape n_samples x n_features.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _squared_euclidean_distances(features: np.ndarray) -> np.ndarray:
    norms = np.sum(features * features, axis=1, keepdims=True)
    return np.maximum(norms + norms.T - 2.0 * (features @ features.T), 0.0)


def _centering_matrix(n_rows: int) -> np.ndarray:
    return np.eye(n_rows, dtype=float) - np.ones((n_rows, n_rows), dtype=float) / float(n_rows)


def _normalize_columns(matrix: np.ndarray) -> np.ndarray:
    vectors = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(vectors, axis=0)
    norms[norms == 0.0] = 1.0
    return vectors / norms[None, :]


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    return 0.5 * (array + array.T)


def _default_estimator():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=13))


def _domain_ids(n_rows: int, source_domains: Sequence[Any] | np.ndarray | None, *, name: str) -> np.ndarray:
    if source_domains is None:
        return np.zeros(n_rows, dtype=int)
    domains = np.asarray(source_domains)
    if domains.shape[0] != n_rows:
        raise ValueError(f"{name} length must match source rows.")
    return domains


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    number = int(value)
    if float(number) != float(value) or number <= 0:
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
        raise ValueError(f"{name} must be a non-negative finite value.")
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a non-negative finite value.")
    return number
