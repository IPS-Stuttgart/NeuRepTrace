"""Riemannian tangent-space transfer utilities for subject-adaptive M/EEG decoding.

The helpers in this module implement label-free and anchor-aware transfer
pipelines for covariance-matrix decoding.  Source labels are used only by the
downstream classifier.  Held-out target recordings may be used to estimate
covariance recentering, stretch/dispersion matching, tangent-space references,
or paired calibration rotations; target labels are intentionally not accepted by
the public prediction APIs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_RIEMANNIAN_EPSILON = 1.0e-9
TANGENT_REFERENCE_SCOPES = ("source", "source_target")
RIEMANNIAN_PROCRUSTES_ROTATION_MODES = ("none", "paired")


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
class RiemannianProcrustesDomainAlignment:
    """Per-source-domain RPA geometry estimated without target labels."""

    source_reference: np.ndarray
    source_dispersion: float
    target_dispersion: float
    stretch: float
    rotation: np.ndarray
    rotation_mode: str
    n_rotation_pairs: int


@dataclass(frozen=True, slots=True)
class RiemannianProcrustesTransferResult:
    """Aligned covariance matrices, tangent features, and RPA provenance."""

    source_features: np.ndarray
    target_features: np.ndarray
    source_covariances: np.ndarray
    target_covariances: np.ndarray
    target_reference: np.ndarray
    target_dispersion: float
    tangent_reference: np.ndarray
    source_domains: np.ndarray
    source_alignments: dict[Any, RiemannianProcrustesDomainAlignment]
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


def normalize_riemannian_procrustes_rotation_mode(value: Any = "none") -> str:
    """Normalize the rotation mode for full Riemannian Procrustes Analysis."""

    normalized = "none" if value is None else str(value).strip().lower().replace("-", "_")
    aliases = {
        "off": "none",
        "false": "none",
        "identity": "none",
        "no_rotation": "none",
        "anchor": "paired",
        "anchors": "paired",
        "paired_anchors": "paired",
        "paired_calibration": "paired",
        "procrustes": "paired",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in RIEMANNIAN_PROCRUSTES_ROTATION_MODES:
        raise ValueError(
            f"rotation_mode must be one of {RIEMANNIAN_PROCRUSTES_ROTATION_MODES}, got {value!r}."
        )
    return normalized


def ensure_spd_matrices(
    matrices: Sequence[np.ndarray] | np.ndarray,
    *,
    epsilon: float = DEFAULT_RIEMANNIAN_EPSILON,
    name: str = "covariances",
) -> np.ndarray:
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


def riemannian_power_spd(
    matrix: np.ndarray,
    power: float,
    *,
    epsilon: float = DEFAULT_RIEMANNIAN_EPSILON,
) -> np.ndarray:
    """Raise an SPD matrix to a Riemannian power in the log-Euclidean chart.

    This is the stretch step used by RPA: after domain recentering, multiplying
    tangent vectors by ``power`` scales each trial's distance from the reference
    without changing the identity reference itself.
    """

    if isinstance(power, (bool, np.bool_)):
        raise ValueError("power must be a positive finite value.")
    exponent = float(power)
    if not np.isfinite(exponent) or exponent <= 0.0:
        raise ValueError("power must be a positive finite value.")
    return _matrix_exp_spd(exponent * _matrix_log_spd(matrix, epsilon=epsilon), epsilon=epsilon)


def riemannian_dispersion(
    matrices: Sequence[np.ndarray] | np.ndarray,
    *,
    reference: np.ndarray | None = None,
    epsilon: float = DEFAULT_RIEMANNIAN_EPSILON,
) -> float:
    """Return RMS affine-log distance of SPD matrices from ``reference``.

    The tangent vectorization scales off-diagonal terms by ``sqrt(2)``, so the
    Euclidean norm of each vector equals the Frobenius norm of the corresponding
    symmetric tangent matrix.
    """

    features, _ = tangent_space_features(matrices, reference=reference, epsilon=epsilon)
    squared_norms = np.sum(np.asarray(features, dtype=float) ** 2, axis=1)
    return float(np.sqrt(np.mean(squared_norms))) if squared_norms.size else 0.0


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


def riemannian_procrustes_transfer_features(
    source_covariances: Sequence[np.ndarray] | np.ndarray,
    target_covariances: Sequence[np.ndarray] | np.ndarray,
    *,
    source_domains: Sequence[Any] | np.ndarray | None = None,
    source_anchor_covariances_by_domain: Mapping[Any, Sequence[np.ndarray] | np.ndarray] | None = None,
    target_anchor_covariances: Sequence[np.ndarray] | np.ndarray | None = None,
    rotation_mode: str = "none",
    tangent_reference_scope: str = "source",
    epsilon: float = DEFAULT_RIEMANNIAN_EPSILON,
) -> RiemannianProcrustesTransferResult:
    """Return tangent features after full Riemannian Procrustes Analysis.

    The RPA pipeline applies the three geometric steps used in the transfer-
    learning literature:

    1. recenter each source domain and the target domain to the identity by
       whitening their log-Euclidean means;
    2. stretch each source domain in the SPD tangent chart so its unlabeled
       dispersion matches the target dispersion;
    3. optionally rotate source-domain tangent vectors with a paired Procrustes
       fit from calibration covariance pairs.

    ``target_covariances`` are unlabeled target adaptation data.  The function
    intentionally has no ``target_labels`` argument.  If paired anchors are built
    from labeled target calibration trials, callers should report that outer
    protocol as Category 3; the geometry here only sees covariance pairs.
    """

    source = ensure_spd_matrices(source_covariances, epsilon=epsilon, name="source_covariances")
    target = ensure_spd_matrices(target_covariances, epsilon=epsilon, name="target_covariances")
    if source.shape[1:] != target.shape[1:]:
        raise ValueError("source_covariances and target_covariances must use the same channel dimensions.")

    rotation = normalize_riemannian_procrustes_rotation_mode(rotation_mode)
    if rotation == "paired" and target_anchor_covariances is None:
        raise ValueError("rotation_mode='paired' requires target_anchor_covariances.")
    if rotation == "none" and (source_anchor_covariances_by_domain is not None or target_anchor_covariances is not None):
        raise ValueError("paired anchor covariances require rotation_mode='paired'.")

    domain_ids = _domain_ids(source.shape[0], source_domains)
    target_aligned, target_reference = align_covariances_to_identity(target, epsilon=epsilon)
    identity = np.eye(target.shape[1], dtype=float)
    target_dispersion = riemannian_dispersion(target_aligned, reference=identity, epsilon=epsilon)
    target_anchor_aligned = None
    if target_anchor_covariances is not None:
        target_anchor = ensure_spd_matrices(
            target_anchor_covariances,
            epsilon=epsilon,
            name="target_anchor_covariances",
        )
        if target_anchor.shape[1:] != target.shape[1:]:
            raise ValueError("target_anchor_covariances must use the same channel dimensions as target_covariances.")
        target_anchor_aligned, _ = align_covariances_to_identity(target_anchor, reference=target_reference, epsilon=epsilon)

    aligned_source = np.empty_like(source)
    alignments: dict[Any, RiemannianProcrustesDomainAlignment] = {}
    for domain in np.unique(domain_ids):
        mask = domain_ids == domain
        domain_source = source[mask]
        centered_source, source_reference = align_covariances_to_identity(domain_source, epsilon=epsilon)
        source_dispersion = riemannian_dispersion(centered_source, reference=identity, epsilon=epsilon)
        stretch = 1.0 if source_dispersion <= epsilon else max(target_dispersion, epsilon) / source_dispersion
        stretched_source = _power_covariances(centered_source, stretch, epsilon=epsilon)

        tangent_rotation = np.eye(_tangent_dimension(source.shape[1]), dtype=float)
        n_rotation_pairs = 0
        if rotation == "paired":
            if source_anchor_covariances_by_domain is None or domain not in source_anchor_covariances_by_domain:
                raise ValueError(f"rotation_mode='paired' requires source anchor covariances for domain {domain!r}.")
            source_anchor = ensure_spd_matrices(
                source_anchor_covariances_by_domain[domain],
                epsilon=epsilon,
                name=f"source_anchor_covariances_by_domain[{domain!r}]",
            )
            if source_anchor.shape[1:] != target.shape[1:]:
                raise ValueError("source anchor covariances must use the same channel dimensions as target_covariances.")
            if target_anchor_aligned is None:  # pragma: no cover - guarded above
                raise ValueError("target_anchor_covariances are required for paired RPA rotation.")
            if source_anchor.shape[0] != target_anchor_aligned.shape[0]:
                raise ValueError(
                    "source and target anchor covariance pairs must have the same row count: "
                    f"{source_anchor.shape[0]} != {target_anchor_aligned.shape[0]}."
                )
            source_anchor_centered, _ = align_covariances_to_identity(source_anchor, reference=source_reference, epsilon=epsilon)
            source_anchor_stretched = _power_covariances(source_anchor_centered, stretch, epsilon=epsilon)
            tangent_rotation = _fit_tangent_space_rotation(
                source_anchor_stretched,
                target_anchor_aligned,
                epsilon=epsilon,
            )
            stretched_source = _apply_tangent_space_rotation(stretched_source, tangent_rotation, epsilon=epsilon)
            n_rotation_pairs = int(source_anchor.shape[0])

        aligned_source[mask] = stretched_source
        alignments[domain] = RiemannianProcrustesDomainAlignment(
            source_reference=source_reference,
            source_dispersion=float(source_dispersion),
            target_dispersion=float(target_dispersion),
            stretch=float(stretch),
            rotation=tangent_rotation,
            rotation_mode=rotation,
            n_rotation_pairs=n_rotation_pairs,
        )

    reference_scope = normalize_tangent_reference_scope(tangent_reference_scope)
    reference_input = aligned_source if reference_scope == "source" else np.concatenate([aligned_source, target_aligned], axis=0)
    tangent_reference = log_euclidean_mean(reference_input, epsilon=epsilon)
    source_features, _ = tangent_space_features(aligned_source, reference=tangent_reference, epsilon=epsilon)
    target_features, _ = tangent_space_features(target_aligned, reference=tangent_reference, epsilon=epsilon)
    return RiemannianProcrustesTransferResult(
        source_features=source_features,
        target_features=target_features,
        source_covariances=aligned_source,
        target_covariances=target_aligned,
        target_reference=target_reference,
        target_dispersion=float(target_dispersion),
        tangent_reference=tangent_reference,
        source_domains=domain_ids,
        source_alignments=alignments,
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


def fit_predict_riemannian_procrustes(
    source_covariances: Sequence[np.ndarray] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    target_covariances: Sequence[np.ndarray] | np.ndarray,
    *,
    source_domains: Sequence[Any] | np.ndarray | None = None,
    source_anchor_covariances_by_domain: Mapping[Any, Sequence[np.ndarray] | np.ndarray] | None = None,
    target_anchor_covariances: Sequence[np.ndarray] | np.ndarray | None = None,
    rotation_mode: str = "none",
    estimator: Any | None = None,
    tangent_reference_scope: str = "source",
    epsilon: float = DEFAULT_RIEMANNIAN_EPSILON,
) -> tuple[Any, RiemannianProcrustesTransferResult, np.ndarray]:
    """Fit a source-label classifier after full RPA and predict target rows."""

    labels = np.asarray(source_labels).reshape(-1)
    transfer = riemannian_procrustes_transfer_features(
        source_covariances,
        target_covariances,
        source_domains=source_domains,
        source_anchor_covariances_by_domain=source_anchor_covariances_by_domain,
        target_anchor_covariances=target_anchor_covariances,
        rotation_mode=rotation_mode,
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


def _power_covariances(matrices: np.ndarray, power: float, *, epsilon: float) -> np.ndarray:
    return np.stack([riemannian_power_spd(matrix, power, epsilon=epsilon) for matrix in matrices], axis=0)


def _tangent_dimension(n_channels: int) -> int:
    return int(n_channels * (n_channels + 1) // 2)


def _unvectorize_symmetric(vector: np.ndarray, n_channels: int, *, scale_off_diagonal: bool = True) -> np.ndarray:
    values = np.asarray(vector, dtype=float).reshape(-1)
    expected = _tangent_dimension(n_channels)
    if values.shape[0] != expected:
        raise ValueError(f"vector length must be {expected} for {n_channels} channels, got {values.shape[0]}.")
    matrix = np.zeros((n_channels, n_channels), dtype=float)
    rows, cols = np.triu_indices(n_channels)
    unscaled = values.copy()
    if scale_off_diagonal:
        unscaled[rows != cols] /= np.sqrt(2.0)
    matrix[rows, cols] = unscaled
    matrix[cols, rows] = unscaled
    return matrix


def _tangent_vectors_at_identity(matrices: np.ndarray, *, epsilon: float) -> np.ndarray:
    return np.vstack([vectorize_symmetric(_matrix_log_spd(matrix, epsilon=epsilon), scale_off_diagonal=True) for matrix in matrices])


def _fit_tangent_space_rotation(
    source_covariances: np.ndarray,
    target_covariances: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    source_vectors = _tangent_vectors_at_identity(source_covariances, epsilon=epsilon)
    target_vectors = _tangent_vectors_at_identity(target_covariances, epsilon=epsilon)
    if source_vectors.shape != target_vectors.shape:
        raise ValueError(
            "source and target tangent anchor matrices must have the same shape: "
            f"{source_vectors.shape} != {target_vectors.shape}."
        )
    if source_vectors.shape[0] < 2:
        raise ValueError("paired RPA rotation requires at least two anchor covariance pairs.")
    u, _singular_values, vt = np.linalg.svd(source_vectors.T @ target_vectors, full_matrices=False)
    return u @ vt


def _apply_tangent_space_rotation(matrices: np.ndarray, rotation: np.ndarray, *, epsilon: float) -> np.ndarray:
    spd = ensure_spd_matrices(matrices, epsilon=epsilon)
    n_channels = spd.shape[1]
    vectors = _tangent_vectors_at_identity(spd, epsilon=epsilon)
    rotation_matrix = np.asarray(rotation, dtype=float)
    if rotation_matrix.shape != (vectors.shape[1], vectors.shape[1]):
        raise ValueError(
            "rotation must be square with tangent-space width "
            f"{vectors.shape[1]}, got {rotation_matrix.shape}."
        )
    rotated = vectors @ rotation_matrix
    return np.stack(
        [
            _matrix_exp_spd(_unvectorize_symmetric(vector, n_channels, scale_off_diagonal=True), epsilon=epsilon)
            for vector in rotated
        ],
        axis=0,
    )


__all__ = [
    "DEFAULT_RIEMANNIAN_EPSILON",
    "RIEMANNIAN_PROCRUSTES_ROTATION_MODES",
    "RiemannianProcrustesDomainAlignment",
    "RiemannianProcrustesTransferResult",
    "RiemannianTransferResult",
    "align_covariances_to_identity",
    "ensure_spd_matrices",
    "fit_predict_riemannian_procrustes",
    "fit_predict_riemannian_transfer",
    "log_euclidean_mean",
    "normalize_riemannian_procrustes_rotation_mode",
    "normalize_tangent_reference_scope",
    "riemannian_dispersion",
    "riemannian_power_spd",
    "riemannian_procrustes_transfer_features",
    "riemannian_tangent_transfer_features",
    "tangent_space_features",
    "vectorize_symmetric",
]
