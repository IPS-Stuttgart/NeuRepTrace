"""Contrastive source-subject alignment utilities.

The contrastive aligner builds a fold-local common template from source anchor
rows only. Rows with the same anchor value across subjects are treated as
positives; different anchor rows are negatives represented by a centered simplex
template. Each source subject receives a ridge projection into that template,
and held-out subjects can be transformed with the averaged source projection or
with a separately fitted target-calibration projection.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from neureptrace.decoding.hyperalignment import ClassAlignment, _feature_matrix, class_alignment_matrices
from neureptrace.decoding.sampling import DEFAULT_CLASS_LIMIT_SEED, DEFAULT_CLASS_LIMIT_SELECTION

CONTRASTIVE_ALIGNMENT_METHOD = "contrastive"
DEFAULT_CONTRASTIVE_REGULARIZATION = 1e-6
_MIN_EIGENVALUE = 1e-12


@dataclass(frozen=True)
class SubjectContrastiveProjection:
    """Linear map from one subject's feature space into the contrastive template."""

    subject_id: Hashable
    feature_mean: np.ndarray
    projection: np.ndarray
    n_alignment_rows: int
    template_mean: np.ndarray


@dataclass(frozen=True)
class ContrastiveAlignmentModel:
    """Fold-local source-subject contrastive alignment model."""

    subject_ids: tuple[Hashable, ...]
    n_components: int
    projections: Mapping[Hashable, SubjectContrastiveProjection]
    template: np.ndarray
    group_feature_mean: np.ndarray | None
    group_projection: np.ndarray | None
    regularization: float

    def transform(self, subject_id: Hashable, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        """Transform rows from a fitted source subject."""

        try:
            projection = self.projections[subject_id]
        except KeyError as exc:
            fitted = ", ".join(str(value) for value in self.subject_ids)
            raise KeyError(f"Unknown contrastive-alignment subject {subject_id!r}. Fitted subjects: {fitted}.") from exc
        return transform_with_contrastive_projection(features, projection)

    def transform_group(
        self,
        features: Sequence[Sequence[float]] | np.ndarray,
        *,
        feature_mean: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Transform an unseen subject with the averaged source projection."""

        if self.group_projection is None or self.group_feature_mean is None:
            raise ValueError("A group contrastive projection is unavailable because source feature dimensions differ.")
        matrix = _feature_matrix(features, name="features")
        if matrix.shape[1] != self.group_projection.shape[0]:
            raise ValueError(
                "features column count does not match the group contrastive projection: "
                f"{matrix.shape[1]} != {self.group_projection.shape[0]}."
            )
        mean = self.group_feature_mean if feature_mean is None else np.asarray(feature_mean, dtype=float).ravel()
        if mean.shape[0] != matrix.shape[1]:
            raise ValueError(f"feature_mean length must match features columns: {mean.shape[0]} != {matrix.shape[1]}.")
        template_mean = np.mean(self.template, axis=0)
        return (matrix - mean) @ self.group_projection + template_mean


def fit_contrastive_alignment(
    features_by_subject: Mapping[Hashable, Sequence[Sequence[float]] | np.ndarray],
    anchors_by_subject: Mapping[Hashable, Sequence[Any] | np.ndarray],
    *,
    sample_mode: str = "class_mean",
    n_repetitions_per_class: int | None = None,
    repetition_selection: str = DEFAULT_CLASS_LIMIT_SELECTION,
    repetition_seed: int | str | None = DEFAULT_CLASS_LIMIT_SEED,
    n_components: int | float = 64,
    regularization: float = DEFAULT_CONTRASTIVE_REGULARIZATION,
) -> tuple[ContrastiveAlignmentModel, ClassAlignment]:
    """Fit source-subject contrastive alignment from row-aligned anchors.

    The source anchors may be decoder classes, stimulus IDs, event codes, or
    repetition-level metadata.  No held-out target data are used here; target use
    is controlled by the caller's target-projection protocol.
    """

    if len(features_by_subject) < 2:
        raise ValueError("Contrastive alignment requires at least two subjects.")
    regularization = _regularization_value(regularization)
    alignment = class_alignment_matrices(
        features_by_subject,
        anchors_by_subject,
        sample_mode=sample_mode,
        n_repetitions_per_class=n_repetitions_per_class,
        repetition_selection=repetition_selection,
        repetition_seed=repetition_seed,
    )
    anchor_matrices = {
        subject_id: _feature_matrix(matrix, name=f"aligned_by_subject[{subject_id!r}]")
        for subject_id, matrix in alignment.aligned_by_subject.items()
    }
    subject_ids = tuple(anchor_matrices)
    n_rows = _common_row_count(anchor_matrices)
    if n_rows < 2:
        raise ValueError("Contrastive alignment requires at least two anchor rows.")
    n_features = _common_feature_count(anchor_matrices)
    actual_components = _actual_component_count(n_components, n_rows=n_rows, n_features=n_features)
    template = contrastive_anchor_template(n_rows, actual_components)
    projections = {
        subject_id: fit_projection_to_contrastive_template(
            anchor_matrices[subject_id],
            template=template,
            regularization=regularization,
            subject_id=subject_id,
        )
        for subject_id in subject_ids
    }
    group_feature_mean, group_projection = _average_projection(projections)
    return (
        ContrastiveAlignmentModel(
            subject_ids=subject_ids,
            n_components=actual_components,
            projections=projections,
            template=template,
            group_feature_mean=group_feature_mean,
            group_projection=group_projection,
            regularization=regularization,
        ),
        alignment,
    )


def contrastive_anchor_template(n_anchor_rows: int, n_components: int | float) -> np.ndarray:
    """Return a centered simplex template for contrastive row anchors."""

    n_rows = int(n_anchor_rows)
    if n_rows < 2:
        raise ValueError("Contrastive anchor templates require at least two rows.")
    actual = _actual_component_count(n_components, n_rows=n_rows, n_features=n_rows)
    centered_identity = np.eye(n_rows, dtype=float) - np.full((n_rows, n_rows), 1.0 / n_rows)
    u, singular_values, _vt = np.linalg.svd(centered_identity, full_matrices=False)
    template = u[:, :actual] * singular_values[:actual]
    template -= np.mean(template, axis=0, keepdims=True)
    return template


def fit_projection_to_contrastive_template(
    features: Sequence[Sequence[float]] | np.ndarray,
    *,
    template: ContrastiveAlignmentModel | Sequence[Sequence[float]] | np.ndarray,
    regularization: float = DEFAULT_CONTRASTIVE_REGULARIZATION,
    subject_id: Hashable = "target",
) -> SubjectContrastiveProjection:
    """Fit one subject's ridge projection to an existing contrastive template."""

    matrix = _feature_matrix(features, name="features")
    template_matrix = _template_matrix(template)
    if matrix.shape[0] != template_matrix.shape[0]:
        raise ValueError(f"features and template need the same row count: {matrix.shape[0]} != {template_matrix.shape[0]}.")
    regularization = _regularization_value(regularization)
    feature_mean = np.mean(matrix, axis=0)
    template_mean = np.mean(template_matrix, axis=0)
    centered_features = matrix - feature_mean
    centered_template = template_matrix - template_mean
    n_features = centered_features.shape[1]
    gram = centered_features.T @ centered_features
    scale = float(np.trace(gram) / max(1, n_features))
    ridge = regularization * max(scale, 1.0)
    projection = np.linalg.solve(gram + ridge * np.eye(n_features), centered_features.T @ centered_template)
    return SubjectContrastiveProjection(
        subject_id=subject_id,
        feature_mean=feature_mean,
        projection=projection,
        n_alignment_rows=matrix.shape[0],
        template_mean=template_mean,
    )


def transform_with_contrastive_projection(
    features: Sequence[Sequence[float]] | np.ndarray,
    projection: SubjectContrastiveProjection,
) -> np.ndarray:
    """Transform rows with a fitted contrastive projection."""

    matrix = _feature_matrix(features, name="features")
    if matrix.shape[1] != projection.projection.shape[0]:
        raise ValueError(
            "features column count does not match the contrastive projection: "
            f"{matrix.shape[1]} != {projection.projection.shape[0]}."
        )
    return (matrix - projection.feature_mean) @ projection.projection + projection.template_mean


def _template_matrix(template: ContrastiveAlignmentModel | Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    if isinstance(template, ContrastiveAlignmentModel):
        return _feature_matrix(template.template, name="template.template")
    return _feature_matrix(template, name="template")


def _average_projection(
    projections: Mapping[Hashable, SubjectContrastiveProjection],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not projections:
        return None, None
    feature_dims = {projection.projection.shape[0] for projection in projections.values()}
    component_dims = {projection.projection.shape[1] for projection in projections.values()}
    if len(feature_dims) != 1 or len(component_dims) != 1:
        return None, None
    feature_mean = np.mean(np.stack([projection.feature_mean for projection in projections.values()], axis=0), axis=0)
    group_projection = np.mean(np.stack([projection.projection for projection in projections.values()], axis=0), axis=0)
    return feature_mean, group_projection


def _common_row_count(matrices: Mapping[Hashable, np.ndarray]) -> int:
    row_counts = {matrix.shape[0] for matrix in matrices.values()}
    if len(row_counts) != 1:
        raise ValueError(f"All contrastive anchor matrices must have the same row count, got {sorted(row_counts)}.")
    return int(next(iter(row_counts)))


def _common_feature_count(matrices: Mapping[Hashable, np.ndarray]) -> int:
    feature_counts = {matrix.shape[1] for matrix in matrices.values()}
    if len(feature_counts) != 1:
        raise ValueError(f"All contrastive anchor matrices must have the same feature count, got {sorted(feature_counts)}.")
    return int(next(iter(feature_counts)))


def _requested_component_count(n_components: int | float) -> int:
    try:
        numeric = float(n_components)
    except (TypeError, ValueError) as exc:
        raise ValueError("n_components must be a positive integer or infinity.") from exc
    if numeric == float("inf"):
        return int(10**12)
    if not np.isfinite(numeric) or numeric % 1.0 != 0.0 or numeric < 1.0:
        raise ValueError("n_components must be a positive integer or infinity.")
    return int(numeric)


def _actual_component_count(n_components: int | float, *, n_rows: int, n_features: int) -> int:
    actual = min(_requested_component_count(n_components), max(0, int(n_rows) - 1), int(n_features))
    if actual < 1:
        raise ValueError(
            "No contrastive alignment components are available. "
            "Use at least two anchor rows and at least one feature."
        )
    return int(actual)


def _regularization_value(value: float) -> float:
    try:
        regularization = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("contrastive regularization must be finite and non-negative.") from exc
    if not np.isfinite(regularization) or regularization < 0.0:
        raise ValueError("contrastive regularization must be finite and non-negative.")
    return max(regularization, _MIN_EIGENVALUE)


__all__ = [
    "CONTRASTIVE_ALIGNMENT_METHOD",
    "DEFAULT_CONTRASTIVE_REGULARIZATION",
    "ContrastiveAlignmentModel",
    "SubjectContrastiveProjection",
    "contrastive_anchor_template",
    "fit_contrastive_alignment",
    "fit_projection_to_contrastive_template",
    "transform_with_contrastive_projection",
]
