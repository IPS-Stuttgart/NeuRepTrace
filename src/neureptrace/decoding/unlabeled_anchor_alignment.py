"""Unlabeled anchor calibration for Category-2 cross-subject alignment.

This module implements the clean deployment version of hyperalignment-like target
calibration: source subjects and the held-out target subject share an external,
label-free calibration axis such as movie time points, stimulus identifiers, or
resting-state segment ids.  Source and target projections are fitted from those
anchors, a source-label classifier can then be trained in the shared latent space,
and target test rows are transformed without using target class labels.

The public API intentionally has no ``target_labels`` argument.  Passing decoded
class labels as anchors would violate the intended Category-2 interpretation and
should be reported as supervised calibration outside this module.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

UNLABELED_ANCHOR_ALIGNMENT_PROTOCOL = "unlabeled_target_anchor_alignment"
UNLABELED_ANCHOR_ALIGNMENT_CATEGORY = "2_unlabeled_target_adaptive"
DEFAULT_UNLABELED_ANCHOR_COMPONENTS = 64
DEFAULT_UNLABELED_ANCHOR_REGULARIZATION = 1e-6
MISSING_ANCHOR_TEXT_VALUES = frozenset({"", "<na>", "<nat>", "na", "n/a", "nan", "nat", "none", "null"})


@dataclass(frozen=True, slots=True)
class AnchorProjection:
    """Linear projection from one subject/domain into an anchor template."""

    domain_id: Hashable
    feature_mean: np.ndarray
    template_mean: np.ndarray
    projection: np.ndarray
    n_anchor_rows: int
    regularization: float


@dataclass(frozen=True, slots=True)
class UnlabeledAnchorAlignmentResult:
    """Aligned source and target features plus Category-2 provenance."""

    train_features: np.ndarray
    test_features: np.ndarray
    source_projections: Mapping[Hashable, AnchorProjection]
    target_projection: AnchorProjection
    common_anchors: tuple[Hashable, ...]
    template: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments,too-many-locals

def fit_unlabeled_anchor_alignment(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_domains: Sequence[Hashable] | np.ndarray,
    source_anchor_values: Sequence[Hashable] | np.ndarray,
    target_calibration_features: Sequence[Sequence[float]] | np.ndarray,
    target_calibration_anchor_values: Sequence[Hashable] | np.ndarray,
    target_test_features: Sequence[Sequence[float]] | np.ndarray,
    n_components: int | str | float = DEFAULT_UNLABELED_ANCHOR_COMPONENTS,
    regularization: float | str = DEFAULT_UNLABELED_ANCHOR_REGULARIZATION,
    min_common_anchors: int | str = 2,
) -> UnlabeledAnchorAlignmentResult:
    """Fit source and target projections from shared unlabeled anchors.

    Parameters
    ----------
    source_features:
        Source feature rows pooled across source subjects/domains.
    source_domains:
        One source-domain identifier per source row, typically subject id.
    source_anchor_values:
        One label-free calibration anchor per source row.  Anchors may be movie
        time-bin ids, stimulus ids, event ids, or other external alignment keys.
        They must not be held-out target class labels if the result is reported as
        Category 2.
    target_calibration_features:
        Held-out target-subject calibration rows used only for fitting the target
        projection.  These rows should be disjoint from scored target test rows
        when the benchmark is not explicitly transductive.
    target_calibration_anchor_values:
        Label-free target calibration anchors in the same anchor space as
        ``source_anchor_values``.
    target_test_features:
        Held-out target-subject rows to transform into the shared anchor space.
    n_components:
        Maximum latent dimension.  The effective dimension is capped by the number
        of common anchors minus one and the feature dimension.
    regularization:
        Ridge regularization for each subject/domain projection.
    min_common_anchors:
        Minimum number of anchors that must be present in every source domain and
        in the target calibration set.

    Returns
    -------
    UnlabeledAnchorAlignmentResult
        Source train rows and target test rows in a shared anchor-template space.

    Notes
    -----
    This is a Category-2 protocol.  It uses source features, source domain ids,
    source anchor values, target calibration features, and target calibration
    anchor values.  It never accepts target class labels.
    """

    source_matrix = _feature_matrix(source_features, name="source_features")
    target_calibration_matrix = _feature_matrix(target_calibration_features, name="target_calibration_features")
    target_test_matrix = _feature_matrix(target_test_features, name="target_test_features")
    if target_calibration_matrix.shape[1] != source_matrix.shape[1]:
        raise ValueError(
            "target_calibration_features and source_features must have the same feature width: "
            f"{target_calibration_matrix.shape[1]} != {source_matrix.shape[1]}."
        )
    if target_test_matrix.shape[1] != source_matrix.shape[1]:
        raise ValueError(
            "target_test_features and source_features must have the same feature width: "
            f"{target_test_matrix.shape[1]} != {source_matrix.shape[1]}."
        )

    source_domain_vector = _hashable_vector(source_domains, expected_length=source_matrix.shape[0], name="source_domains")
    source_anchor_vector = _hashable_vector(source_anchor_values, expected_length=source_matrix.shape[0], name="source_anchor_values", allow_missing=True)
    target_anchor_vector = _hashable_vector(
        target_calibration_anchor_values,
        expected_length=target_calibration_matrix.shape[0],
        name="target_calibration_anchor_values",
        allow_missing=True,
    )
    reg = _normalize_nonnegative_float(regularization, name="regularization")
    min_anchors = _normalize_positive_int(min_common_anchors, name="min_common_anchors")

    source_domains_ordered = tuple(dict.fromkeys(source_domain_vector.tolist()))
    if len(source_domains_ordered) < 1:
        raise ValueError("At least one source domain is required.")
    source_anchor_means = {
        domain: _anchor_means(source_matrix[source_domain_vector == domain], source_anchor_vector[source_domain_vector == domain])
        for domain in source_domains_ordered
    }
    target_anchor_means = _anchor_means(target_calibration_matrix, target_anchor_vector)
    common_anchors = _common_anchor_order(source_anchor_means, target_anchor_means, source_anchor_vector, target_anchor_vector)
    if len(common_anchors) < min_anchors:
        raise ValueError(
            "unlabeled anchor alignment requires at least "
            f"{min_anchors} common anchors across all source domains and target calibration; got {len(common_anchors)}."
        )

    source_anchor_matrices = {
        domain: _matrix_for_anchors(anchor_means, common_anchors, name=f"source domain {domain!r}")
        for domain, anchor_means in source_anchor_means.items()
    }
    target_anchor_matrix = _matrix_for_anchors(target_anchor_means, common_anchors, name="target calibration")
    template = anchor_template(len(common_anchors), n_components=n_components, feature_dim=source_matrix.shape[1])

    source_projections = {
        domain: fit_anchor_projection(anchor_matrix, template, domain_id=domain, regularization=reg)
        for domain, anchor_matrix in source_anchor_matrices.items()
    }
    target_projection = fit_anchor_projection(target_anchor_matrix, template, domain_id="target", regularization=reg)

    train_aligned = np.empty((source_matrix.shape[0], template.shape[1]), dtype=float)
    for domain, projection in source_projections.items():
        mask = source_domain_vector == domain
        train_aligned[mask] = transform_with_anchor_projection(source_matrix[mask], projection)
    test_aligned = transform_with_anchor_projection(target_test_matrix, target_projection)

    metadata = _metadata(
        n_source_rows=source_matrix.shape[0],
        n_target_calibration_rows=target_calibration_matrix.shape[0],
        n_target_test_rows=target_test_matrix.shape[0],
        feature_dim=source_matrix.shape[1],
        latent_dim=template.shape[1],
        n_source_domains=len(source_domains_ordered),
        n_common_anchors=len(common_anchors),
        common_anchors=common_anchors,
        regularization=reg,
        requested_components=n_components,
        min_common_anchors=min_anchors,
    )
    return UnlabeledAnchorAlignmentResult(
        train_features=train_aligned.astype(np.float32, copy=False),
        test_features=test_aligned.astype(np.float32, copy=False),
        source_projections=source_projections,
        target_projection=target_projection,
        common_anchors=common_anchors,
        template=template.astype(np.float32, copy=False),
        metadata=metadata,
    )


def anchor_template(n_anchor_rows: int | str, *, n_components: int | str | float = DEFAULT_UNLABELED_ANCHOR_COMPONENTS, feature_dim: int | str | None = None) -> np.ndarray:
    """Return a centered simplex template for shared unlabeled anchors."""

    n_rows = _normalize_positive_int(n_anchor_rows, name="n_anchor_rows")
    if n_rows < 2:
        raise ValueError("At least two anchor rows are required to build an anchor template.")
    feature_cap = n_rows if feature_dim is None else _normalize_positive_int(feature_dim, name="feature_dim")
    actual = _effective_components(n_components, n_anchor_rows=n_rows, feature_dim=feature_cap)
    centered_identity = np.eye(n_rows, dtype=float) - np.full((n_rows, n_rows), 1.0 / n_rows)
    u, singular_values, _vt = np.linalg.svd(centered_identity, full_matrices=False)
    template = u[:, :actual] * singular_values[:actual]
    template -= np.mean(template, axis=0, keepdims=True)
    return template


def fit_anchor_projection(
    anchor_features: Sequence[Sequence[float]] | np.ndarray,
    template: Sequence[Sequence[float]] | np.ndarray,
    *,
    domain_id: Hashable = "domain",
    regularization: float | str = DEFAULT_UNLABELED_ANCHOR_REGULARIZATION,
) -> AnchorProjection:
    """Fit a ridge projection from one domain's anchor rows to a template."""

    matrix = _feature_matrix(anchor_features, name="anchor_features")
    template_matrix = _feature_matrix(template, name="template")
    if matrix.shape[0] != template_matrix.shape[0]:
        raise ValueError(f"anchor_features and template must have the same row count: {matrix.shape[0]} != {template_matrix.shape[0]}.")
    reg = _normalize_nonnegative_float(regularization, name="regularization")
    try:
        hash(domain_id)
    except TypeError as exc:
        raise ValueError(f"domain_id must be hashable, got {domain_id!r}.") from exc

    feature_mean = np.mean(matrix, axis=0)
    template_mean = np.mean(template_matrix, axis=0)
    centered_features = matrix - feature_mean
    centered_template = template_matrix - template_mean
    gram = centered_features.T @ centered_features
    scale = float(np.trace(gram) / max(1, gram.shape[0]))
    ridge = reg * max(scale, 1.0)
    system = gram + ridge * np.eye(gram.shape[0], dtype=float)
    rhs = centered_features.T @ centered_template
    try:
        projection = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        projection = np.linalg.pinv(system) @ rhs
    return AnchorProjection(
        domain_id=domain_id,
        feature_mean=feature_mean,
        template_mean=template_mean,
        projection=projection,
        n_anchor_rows=matrix.shape[0],
        regularization=reg,
    )


def transform_with_anchor_projection(features: Sequence[Sequence[float]] | np.ndarray, projection: AnchorProjection) -> np.ndarray:
    """Transform rows with a fitted anchor projection."""

    matrix = _feature_matrix(features, name="features")
    if matrix.shape[1] != projection.projection.shape[0]:
        raise ValueError(
            "features column count does not match the anchor projection: "
            f"{matrix.shape[1]} != {projection.projection.shape[0]}."
        )
    return (matrix - projection.feature_mean) @ projection.projection + projection.template_mean


def _anchor_means(features: np.ndarray, anchors: np.ndarray) -> dict[Hashable, np.ndarray]:
    sums: dict[Hashable, np.ndarray] = {}
    counts: dict[Hashable, int] = {}
    for row, anchor in zip(features, anchors, strict=True):
        if _is_missing_anchor(anchor):
            continue
        try:
            hash(anchor)
        except TypeError as exc:
            raise ValueError(f"Anchor values must be hashable; got {anchor!r}.") from exc
        if anchor not in sums:
            sums[anchor] = np.zeros(features.shape[1], dtype=float)
            counts[anchor] = 0
        sums[anchor] += np.asarray(row, dtype=float)
        counts[anchor] += 1
    if not sums:
        raise ValueError("At least one non-missing anchor value is required.")
    return {anchor: sums[anchor] / float(counts[anchor]) for anchor in sums}


def _common_anchor_order(
    source_anchor_means: Mapping[Hashable, Mapping[Hashable, np.ndarray]],
    target_anchor_means: Mapping[Hashable, np.ndarray],
    source_anchor_values: np.ndarray,
    target_anchor_values: np.ndarray,
) -> tuple[Hashable, ...]:
    common: set[Hashable] = set(target_anchor_means)
    for anchor_means in source_anchor_means.values():
        common &= set(anchor_means)
    ordered_candidates = []
    for anchor in (*target_anchor_values.tolist(), *source_anchor_values.tolist()):
        if _is_missing_anchor(anchor) or anchor not in common or anchor in ordered_candidates:
            continue
        ordered_candidates.append(anchor)
    return tuple(ordered_candidates)


def _matrix_for_anchors(anchor_means: Mapping[Hashable, np.ndarray], anchors: Sequence[Hashable], *, name: str) -> np.ndarray:
    missing = [anchor for anchor in anchors if anchor not in anchor_means]
    if missing:
        preview = ", ".join(repr(anchor) for anchor in missing[:5])
        raise ValueError(f"{name} is missing common anchor rows: {preview}.")
    return np.vstack([anchor_means[anchor] for anchor in anchors])


def _metadata(
    *,
    n_source_rows: int,
    n_target_calibration_rows: int,
    n_target_test_rows: int,
    feature_dim: int,
    latent_dim: int,
    n_source_domains: int,
    n_common_anchors: int,
    common_anchors: tuple[Hashable, ...],
    regularization: float,
    requested_components: int | str | float,
    min_common_anchors: int,
) -> dict[str, Any]:
    return {
        "unlabeled_anchor_alignment": True,
        "unlabeled_anchor_alignment_protocol": UNLABELED_ANCHOR_ALIGNMENT_PROTOCOL,
        "unlabeled_anchor_alignment_category": UNLABELED_ANCHOR_ALIGNMENT_CATEGORY,
        "unlabeled_anchor_alignment_uses_source_features": True,
        "unlabeled_anchor_alignment_uses_source_domains": True,
        "unlabeled_anchor_alignment_uses_source_anchor_values": True,
        "unlabeled_anchor_alignment_uses_target_calibration_features": True,
        "unlabeled_anchor_alignment_uses_target_anchor_values": True,
        "unlabeled_anchor_alignment_uses_target_labels": False,
        "unlabeled_anchor_alignment_valid_for_strict_source_only": False,
        "unlabeled_anchor_alignment_valid_for_unlabeled_target_adaptation": True,
        "unlabeled_anchor_alignment_valid_for_protocol2_benchmark": True,
        "unlabeled_anchor_alignment_valid_for_oracle_upper_bound": False,
        "unlabeled_anchor_alignment_note": "target projection fit from label-free calibration anchors; do not pass decoded class labels as anchors for Category-2 claims",
        "unlabeled_anchor_alignment_n_source_rows": int(n_source_rows),
        "unlabeled_anchor_alignment_n_target_calibration_rows": int(n_target_calibration_rows),
        "unlabeled_anchor_alignment_n_target_test_rows": int(n_target_test_rows),
        "unlabeled_anchor_alignment_feature_dim": int(feature_dim),
        "unlabeled_anchor_alignment_latent_dim": int(latent_dim),
        "unlabeled_anchor_alignment_n_source_domains": int(n_source_domains),
        "unlabeled_anchor_alignment_n_common_anchors": int(n_common_anchors),
        "unlabeled_anchor_alignment_common_anchor_preview": "|".join(str(anchor) for anchor in common_anchors[:12]),
        "unlabeled_anchor_alignment_regularization": float(regularization),
        "unlabeled_anchor_alignment_requested_components": str(requested_components),
        "unlabeled_anchor_alignment_min_common_anchors": int(min_common_anchors),
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


def _hashable_vector(values: Sequence[Hashable] | np.ndarray, *, expected_length: int, name: str, allow_missing: bool = False) -> np.ndarray:
    vector = np.asarray(values, dtype=object).reshape(-1)
    if vector.shape[0] != expected_length:
        raise ValueError(f"{name} must contain one value per row: {vector.shape[0]} != {expected_length}.")
    for value in vector.tolist():
        if allow_missing and _is_missing_anchor(value):
            continue
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError(f"{name} values must be hashable; got {value!r}.") from exc
    return vector


def _is_missing_anchor(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in MISSING_ANCHOR_TEXT_VALUES:
        return True
    return False


def _effective_components(n_components: int | str | float, *, n_anchor_rows: int, feature_dim: int) -> int:
    maximum = max(1, min(int(n_anchor_rows) - 1, int(feature_dim)))
    if isinstance(n_components, str):
        text = n_components.strip().lower()
        if text in {"", "default"}:
            requested: int | float = DEFAULT_UNLABELED_ANCHOR_COMPONENTS
        elif text in {"all", "full", "inf", "infinity"}:
            requested = float("inf")
        else:
            requested = float(text)
    else:
        requested = float(n_components)
    if requested == float("inf"):
        return maximum
    if not np.isfinite(requested) or requested % 1.0 != 0.0 or requested < 1.0:
        raise ValueError("n_components must be a positive integer, 'all', or infinity.")
    return min(int(requested), maximum)


def _normalize_positive_int(value: int | str, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if not np.isfinite(numeric) or numeric % 1.0 != 0.0 or numeric < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(numeric)


def _normalize_nonnegative_float(value: float | str, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and non-negative.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and non-negative.") from exc
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return parsed
