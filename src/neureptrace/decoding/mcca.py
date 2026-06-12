"""Multiset CCA alignment utilities for cross-subject feature decoding.

The implementation follows the common MAXVAR/SUMCOR-style recipe used for
multi-view alignment:

1. each subject's aligned samples are centered and whitened with a thin SVD;
2. the whitened subject matrices are concatenated across feature blocks;
3. an SVD of the concatenated matrix defines shared canonical axes;
4. subject-specific projections map raw features into the shared M-CCA space.

The API deliberately separates the generic linear alignment from any particular
M/EEG dataset convention. Dataset-specific projects should provide aligned rows,
for example class prototypes or class/repetition samples with the same row order
for every subject.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from neureptrace.decoding.sampling import (
    DEFAULT_CLASS_LIMIT_SEED,
    DEFAULT_CLASS_LIMIT_SELECTION,
    normalize_class_limit_seed,
    normalize_class_limit_selection,
    select_class_limited_indices,
)

CLASS_ALIGNMENT_SAMPLE_MODES = ("class_mean", "class_repetition")


@dataclass(frozen=True)
class SubjectMCCAProjection:
    """Subject-specific linear map into an M-CCA common space."""

    subject_id: Hashable
    feature_mean: np.ndarray
    prewhitener: np.ndarray
    projection: np.ndarray
    rank: int
    n_alignment_rows: int


@dataclass(frozen=True)
class MCCAModel:
    """Fitted M-CCA model with one projection per fitted subject.

    ``projection`` matrices have shape ``n_features x n_components`` and can be
    applied to trial-level feature matrices from the matching subject. The group
    projection is a calibration-free fallback for a new subject with the same raw
    feature layout; it is useful as a baseline but is not a replacement for a
    target-subject M-CCA projection estimated from calibration samples.
    """

    subject_ids: tuple[Hashable, ...]
    n_components: int
    regularization: float
    projections: Mapping[Hashable, SubjectMCCAProjection]
    component_scores: np.ndarray
    singular_values: np.ndarray
    explained_variance_ratio: np.ndarray
    group_feature_mean: np.ndarray | None
    group_projection: np.ndarray | None
    normalize_components: bool

    def transform(self, subject_id: Hashable, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        """Transform rows from a fitted subject into the common M-CCA space."""

        try:
            subject_projection = self.projections[subject_id]
        except KeyError as exc:
            fitted = ", ".join(str(value) for value in self.subject_ids)
            raise KeyError(f"Unknown M-CCA subject {subject_id!r}. Fitted subjects: {fitted}.") from exc
        matrix = _feature_matrix(features, name="features")
        if matrix.shape[1] != subject_projection.projection.shape[0]:
            raise ValueError(
                "features column count does not match the fitted subject projection: "
                f"{matrix.shape[1]} != {subject_projection.projection.shape[0]}."
            )
        return (matrix - subject_projection.feature_mean) @ subject_projection.projection

    def transform_group(
        self,
        features: Sequence[Sequence[float]] | np.ndarray,
        *,
        feature_mean: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Transform rows with the across-training-subject average projection.

        This is intended for calibration-free transfer to a subject not included
        in ``fit_mcca``. Passing ``feature_mean`` allows an unsupervised centering
        estimate from the target subject; when omitted the average training
        subject alignment mean is used.
        """

        if self.group_projection is None or self.group_feature_mean is None:
            raise ValueError("A group projection is unavailable because fitted subjects have incompatible feature dimensions.")
        matrix = _feature_matrix(features, name="features")
        if matrix.shape[1] != self.group_projection.shape[0]:
            raise ValueError(
                "features column count does not match the group projection: "
                f"{matrix.shape[1]} != {self.group_projection.shape[0]}."
            )
        if feature_mean is None:
            mean = self.group_feature_mean
        else:
            mean = np.asarray(feature_mean, dtype=float).ravel()
            if mean.shape[0] != matrix.shape[1]:
                raise ValueError(f"feature_mean length must match features columns: {mean.shape[0]} != {matrix.shape[1]}.")
        return (matrix - mean) @ self.group_projection


@dataclass(frozen=True)
class ClassAlignment:
    """Aligned rows built from class labels for a collection of subjects."""

    aligned_by_subject: Mapping[Hashable, np.ndarray]
    classes: np.ndarray
    sample_mode: str
    n_repetitions_per_class: int | None
    repetition_selection: str | None = None
    repetition_seed: int | None = None
    repetition_offsets_by_class: Mapping[int, np.ndarray] | None = None

    @property
    def n_alignment_rows(self) -> int:
        """Number of row anchors supplied to the common-space fit."""

        row_counts = {np.asarray(matrix).shape[0] for matrix in self.aligned_by_subject.values()}
        if not row_counts:
            return 0
        if len(row_counts) != 1:
            raise ValueError(f"All subject alignment matrices must have the same row count, got {sorted(row_counts)}.")
        return int(next(iter(row_counts)))

    @property
    def n_classes(self) -> int:
        """Number of class labels used to build the alignment anchors."""

        return int(np.asarray(self.classes).size)

    @property
    def max_centered_rank(self) -> int:
        """Maximum rank available after centering the alignment anchors."""

        return max(self.n_alignment_rows - 1, 0)

    @property
    def low_rank_warning(self) -> str | None:
        """Human-readable warning for class-mean anchor rank collapse."""

        if self.sample_mode == "class_mean" and self.n_classes <= 3:
            return (
                f"class_mean alignment has only {self.n_classes} class anchors; "
                f"after centering, at most {self.max_centered_rank} common-space components are identifiable. "
                "Use richer anchors such as class_repetition, pseudotrials, stimulus identity, "
                "or target calibration before concluding alignment is ineffective."
            )
        return None

    @property
    def selected_offsets_by_class(self) -> Mapping[int, np.ndarray] | None:
        """Backward-compatible alias for stored class-repetition offsets."""

        return self.repetition_offsets_by_class


# pylint: disable-next=too-many-locals
def fit_mcca(
    aligned_by_subject: Mapping[Hashable, Sequence[Sequence[float]] | np.ndarray],
    *,
    n_components: int | float = 64,
    regularization: float = 1e-6,
    subject_pca_components: int | float | None = None,
    rank_tolerance: float = 1e-10,
    normalize_components: bool = True,
) -> MCCAModel:
    """Fit a multiset CCA projection from row-aligned subject matrices.

    Parameters
    ----------
    aligned_by_subject:
        Mapping from subject id to a matrix with shape
        ``n_aligned_samples x n_features``. All matrices must have the same
        number of rows and the same row order.
    n_components:
        Number of common M-CCA components to retain. The actual number is capped
        by the available rank of the concatenated whitened matrices.
    regularization:
        Non-negative ridge term added to each subject covariance eigenvalue
        before whitening. This stabilizes high-dimensional MEG features.
    subject_pca_components:
        Optional cap on the thin within-subject whitening rank before multiset
        alignment. ``None`` keeps all numerically nonzero subject components.
    rank_tolerance:
        Minimum covariance eigenvalue retained during subject whitening.
    normalize_components:
        When true, rescale columns so pooled aligned projected samples have unit
        standard deviation. This helps downstream classifiers see comparable
        feature scales.
    """

    if len(aligned_by_subject) < 2:
        raise ValueError("M-CCA requires at least two subjects.")
    if regularization < 0:
        raise ValueError("regularization must be non-negative.")

    subject_ids = tuple(aligned_by_subject.keys())
    matrices = {subject_id: _feature_matrix(matrix, name=f"aligned_by_subject[{subject_id!r}]") for subject_id, matrix in aligned_by_subject.items()}
    n_rows = _check_common_alignment_rows(matrices)
    if n_rows < 2:
        raise ValueError("M-CCA requires at least two aligned rows per subject.")

    means: dict[Hashable, np.ndarray] = {}
    prewhiteners: dict[Hashable, np.ndarray] = {}
    whitened_blocks: list[np.ndarray] = []
    ranks: dict[Hashable, int] = {}
    for subject_id in subject_ids:
        mean, prewhitener, whitened = _fit_subject_prewhitener(
            matrices[subject_id],
            subject_id=subject_id,
            regularization=regularization,
            subject_pca_components=subject_pca_components,
            rank_tolerance=rank_tolerance,
        )
        means[subject_id] = mean
        prewhiteners[subject_id] = prewhitener
        ranks[subject_id] = int(prewhitener.shape[1])
        whitened_blocks.append(whitened)

    concatenated = np.hstack(whitened_blocks)
    concatenated = concatenated - np.mean(concatenated, axis=0, keepdims=True)
    _left, singular_values, right_t = np.linalg.svd(concatenated, full_matrices=False)

    # Centering row-aligned matrices caps the identifiable common-space rank.
    shared_rank = _numerical_svd_rank(singular_values, rank_tolerance=rank_tolerance)
    requested_components = _requested_component_count(n_components)
    actual_components = min(requested_components, shared_rank)
    if actual_components < 1:
        raise ValueError("No M-CCA components are available after the shared-space SVD.")

    component_vectors = right_t.T[:, :actual_components]
    projections = _subject_projections_from_blocks(
        subject_ids,
        prewhiteners,
        ranks,
        component_vectors,
        n_components=actual_components,
        n_alignment_rows=n_rows,
        means=means,
    )
    if normalize_components:
        projections = _rescale_subject_projections(matrices, projections)

    component_scores = np.mean(
        np.stack([_transform_with_projection(matrices[subject_id], projections[subject_id]) for subject_id in subject_ids], axis=0),
        axis=0,
    )
    group_feature_mean, group_projection = _average_projection(
        projections,
        matrices=matrices if normalize_components else None,
    )
    explained = _explained_variance_ratio(singular_values)
    return MCCAModel(
        subject_ids=subject_ids,
        n_components=actual_components,
        regularization=float(regularization),
        projections=projections,
        component_scores=component_scores,
        singular_values=singular_values[:actual_components],
        explained_variance_ratio=explained[:actual_components],
        group_feature_mean=group_feature_mean,
        group_projection=group_projection,
        normalize_components=bool(normalize_components),
    )


def _numerical_svd_rank(singular_values: np.ndarray, *, rank_tolerance: float) -> int:
    values = np.asarray(singular_values, dtype=float).ravel()
    if values.size == 0:
        return 0
    scale = max(float(np.max(values)), 1.0)
    threshold = max(float(rank_tolerance), float(rank_tolerance) * scale)
    return int(np.sum(values > threshold))


def class_alignment_matrices(
    features_by_subject: Mapping[Hashable, Sequence[Sequence[float]] | np.ndarray],
    labels_by_subject: Mapping[Hashable, Sequence | np.ndarray],
    *,
    sample_mode: str = "class_mean",
    n_repetitions_per_class: int | None = None,
    repetition_selection: str = DEFAULT_CLASS_LIMIT_SELECTION,
    repetition_seed: int | str | None = DEFAULT_CLASS_LIMIT_SEED,
) -> ClassAlignment:
    """Build row-aligned matrices from class-labeled subject feature matrices.

    ``class_mean`` uses one class-average row per class and subject.
    ``class_repetition`` uses one row per class and within-class repetition index;
    this preserves more sample-level structure and is useful when every subject
    has repeated presentations of the same stimulus set. Repetition caps are
    sampled reproducibly by default instead of taking the earliest rows in each
    class, avoiding run/order confounds in ordered datasets.
    """

    _check_subject_keys(features_by_subject, labels_by_subject)
    sample_mode = _normalize_sample_mode(sample_mode)
    if n_repetitions_per_class is not None and n_repetitions_per_class < 1:
        raise ValueError("n_repetitions_per_class must be positive or None.")

    subject_ids = tuple(features_by_subject.keys())
    features = {subject_id: _feature_matrix(matrix, name=f"features_by_subject[{subject_id!r}]") for subject_id, matrix in features_by_subject.items()}
    labels = {
        subject_id: _label_vector(labels_by_subject[subject_id], expected_length=features[subject_id].shape[0], name=f"labels_by_subject[{subject_id!r}]")
        for subject_id in subject_ids
    }
    classes = _common_classes(labels)
    if sample_mode == "class_mean":
        aligned = {subject_id: _class_mean_matrix(features[subject_id], labels[subject_id], classes) for subject_id in subject_ids}
        repetitions = None
        normalized_selection = None
        normalized_seed = None
        repetition_offsets = None
    else:
        repetitions = _common_repetition_count(labels, classes, requested=n_repetitions_per_class)
        normalized_selection = normalize_class_limit_selection(repetition_selection)
        normalized_seed = normalize_class_limit_seed(repetition_seed)
        repetition_offsets = _common_repetition_offsets(
            labels,
            classes,
            repetitions,
            selection=normalized_selection,
            seed=normalized_seed,
        )
        aligned = {
            subject_id: _class_repetition_matrix(
                features[subject_id],
                labels[subject_id],
                classes,
                repetitions,
                selection=normalized_selection,
                seed=normalized_seed,
                selected_offsets_by_class=repetition_offsets,
            )
            for subject_id in subject_ids
        }

    return ClassAlignment(
        aligned_by_subject=aligned,
        classes=classes,
        sample_mode=sample_mode,
        n_repetitions_per_class=repetitions,
        repetition_selection=normalized_selection,
        repetition_seed=normalized_seed,
        repetition_offsets_by_class=repetition_offsets,
    )


def fit_class_mcca(
    features_by_subject: Mapping[Hashable, Sequence[Sequence[float]] | np.ndarray],
    labels_by_subject: Mapping[Hashable, Sequence | np.ndarray],
    *,
    sample_mode: str = "class_mean",
    n_repetitions_per_class: int | None = None,
    repetition_selection: str = DEFAULT_CLASS_LIMIT_SELECTION,
    repetition_seed: int | str | None = DEFAULT_CLASS_LIMIT_SEED,
    n_components: int | float = 64,
    regularization: float = 1e-6,
    subject_pca_components: int | float | None = None,
    rank_tolerance: float = 1e-10,
    normalize_components: bool = True,
) -> tuple[MCCAModel, ClassAlignment]:
    """Fit M-CCA using aligned rows derived from class labels."""

    alignment = class_alignment_matrices(
        features_by_subject,
        labels_by_subject,
        sample_mode=sample_mode,
        n_repetitions_per_class=n_repetitions_per_class,
        repetition_selection=repetition_selection,
        repetition_seed=repetition_seed,
    )
    model = fit_mcca(
        alignment.aligned_by_subject,
        n_components=n_components,
        regularization=regularization,
        subject_pca_components=subject_pca_components,
        rank_tolerance=rank_tolerance,
        normalize_components=normalize_components,
    )
    return model, alignment


def _fit_subject_prewhitener(
    matrix: np.ndarray,
    *,
    subject_id: Hashable | None = None,
    regularization: float,
    subject_pca_components: int | float | None,
    rank_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(matrix, axis=0)
    centered = matrix - mean
    _u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    if singular_values.size == 0:
        raise ValueError("Subject alignment matrix has no singular values.")
    eigenvalues = (singular_values**2) / max(centered.shape[0] - 1, 1)
    keep = eigenvalues > rank_tolerance
    if subject_pca_components is not None and subject_pca_components != float("inf"):
        max_rank = int(subject_pca_components)
        if max_rank < 1:
            raise ValueError("subject_pca_components must be positive, infinity, or None.")
        keep_indices = np.flatnonzero(keep)[:max_rank]
    else:
        keep_indices = np.flatnonzero(keep)
    if keep_indices.size == 0:
        context = "" if subject_id is None else f" for subject {subject_id!r}"
        raise ValueError(
            f"Subject alignment matrix{context} has no retained centered components. "
            "The alignment anchors are rank deficient after centering; use richer anchors or lower rank_tolerance."
        )
    components = vt[keep_indices].T
    scales = 1.0 / np.sqrt(eigenvalues[keep_indices] + regularization)
    prewhitener = components * scales[None, :]
    whitened = centered @ prewhitener
    return mean, prewhitener, whitened


def _subject_projections_from_blocks(
    subject_ids,
    prewhiteners,
    ranks,
    component_vectors,
    *,
    n_components,
    n_alignment_rows,
    means,
):
    projections = {}
    start = 0
    scale = np.sqrt(max(n_alignment_rows - 1, 1))
    for subject_id in subject_ids:
        stop = start + ranks[subject_id]
        subject_weights = component_vectors[start:stop, :n_components]
        projection = prewhiteners[subject_id] @ subject_weights * scale
        projections[subject_id] = SubjectMCCAProjection(
            subject_id=subject_id,
            feature_mean=means[subject_id],
            prewhitener=prewhiteners[subject_id],
            projection=projection,
            rank=ranks[subject_id],
            n_alignment_rows=n_alignment_rows,
        )
        start = stop
    return projections


def _rescale_subject_projections(matrices, projections):
    transformed = np.vstack([_transform_with_projection(matrices[subject_id], projection) for subject_id, projection in projections.items()])
    scale = np.std(transformed, axis=0, ddof=1)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return {
        subject_id: SubjectMCCAProjection(
            subject_id=projection.subject_id,
            feature_mean=projection.feature_mean,
            prewhitener=projection.prewhitener,
            projection=projection.projection / scale[None, :],
            rank=projection.rank,
            n_alignment_rows=projection.n_alignment_rows,
        )
        for subject_id, projection in projections.items()
    }


def _transform_with_projection(features: np.ndarray, projection: SubjectMCCAProjection) -> np.ndarray:
    return (features - projection.feature_mean) @ projection.projection


def _average_projection(projections, *, matrices=None):
    feature_dims = {projection.projection.shape[0] for projection in projections.values()}
    if len(feature_dims) != 1:
        return None, None
    mean = np.mean(np.stack([projection.feature_mean for projection in projections.values()], axis=0), axis=0)
    matrix = np.mean(np.stack([projection.projection for projection in projections.values()], axis=0), axis=0)
    if matrices is not None:
        matrix = _rescale_group_projection(matrix, projections, matrices)
    return mean, matrix


def _rescale_group_projection(matrix: np.ndarray, projections, matrices) -> np.ndarray:
    """Scale the source-average fallback projection on subject-centered data.

    The group projection is a calibration-free fallback for unseen subjects, but
    its scale is estimated from fitted source subjects.  Source subjects must be
    centered with their own fitted alignment means when estimating this scale.
    Centering every source matrix with the across-source average mean introduces
    between-subject offsets into the variance estimate; with large MEG baseline
    offsets this can shrink the fallback projection and make source-only M-CCA
    look worse than it is.
    """
    transformed = np.vstack(
        [
            (
                _feature_matrix(matrices[subject_id], name=f"matrices[{subject_id!r}]")
                - projections[subject_id].feature_mean
            )
            @ matrix
            for subject_id in matrices
        ]
    )
    scale = np.std(transformed, axis=0, ddof=1)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return matrix / scale[None, :]


def _class_mean_matrix(features: np.ndarray, labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    return np.vstack([np.mean(features[_label_mask(labels, class_label)], axis=0) for class_label in classes])


def _class_repetition_matrix(
    features: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    repetitions: int,
    *,
    selection: str = DEFAULT_CLASS_LIMIT_SELECTION,
    seed: int | str | None = DEFAULT_CLASS_LIMIT_SEED,
    selected_offsets_by_class: Mapping[int, Sequence[int] | np.ndarray] | None = None,
) -> np.ndarray:
    rows = []
    for class_position, class_label in enumerate(classes):
        class_features = features[_label_mask(labels, class_label)]
        if class_features.shape[0] < repetitions:
            raise ValueError(f"Class {class_label!r} has only {class_features.shape[0]} repetitions, need {repetitions}.")
        if selected_offsets_by_class is None:
            selected = select_class_limited_indices(
                np.zeros(class_features.shape[0], dtype=int),
                repetitions,
                selection=selection,
                seed=seed,
                seed_context=class_position,
            )
        else:
            selected = np.asarray(selected_offsets_by_class[class_position], dtype=int)
            if selected.ndim != 1:
                raise ValueError("selected repetition offsets must be one-dimensional.")
            if selected.size != repetitions:
                raise ValueError(f"selected repetition offsets must contain {repetitions} entries, got {selected.size}.")
            if selected.size and (int(np.min(selected)) < 0 or int(np.max(selected)) >= class_features.shape[0]):
                raise ValueError(f"selected repetition offsets for class {class_label!r} are outside the available repetitions.")
        rows.extend(class_features[selected])
    return np.vstack(rows)


def _common_repetition_offsets(
    labels_by_subject: Mapping[Hashable, np.ndarray],
    classes: np.ndarray,
    repetitions: int,
    *,
    selection: str,
    seed: int | str | None,
) -> dict[int, np.ndarray]:
    """Sample one common set of within-class offsets for all subjects."""

    offsets = {}
    for class_position, class_label in enumerate(classes):
        available = min(_count_label(labels, class_label) for labels in labels_by_subject.values())
        if available < repetitions:
            raise ValueError(f"Class {class_label!r} has only {available} common repetitions, need {repetitions}.")
        offsets[class_position] = select_class_limited_indices(
            np.zeros(available, dtype=int),
            repetitions,
            selection=selection,
            seed=seed,
            seed_context=class_position,
        )
    return offsets


def _common_classes(labels_by_subject: Mapping[Hashable, np.ndarray]) -> np.ndarray:
    subject_ids = tuple(labels_by_subject.keys())
    first_classes = _ordered_unique_labels(labels_by_subject[subject_ids[0]])
    for subject_id in subject_ids[1:]:
        classes = _ordered_unique_labels(labels_by_subject[subject_id])
        if not _same_label_set(first_classes, classes):
            raise ValueError(f"Subject {subject_id!r} has classes {classes.tolist()}, expected {first_classes.tolist()}.")
    return first_classes


def _ordered_unique_labels(labels: Sequence | np.ndarray) -> np.ndarray:
    """Return unique labels in first-observed order without sorting."""

    values = np.asarray(labels, dtype=object).reshape(-1)
    unique: list[object] = []
    for value in values:
        if not _contains_label(unique, value):
            unique.append(value)
    out = np.empty(len(unique), dtype=object)
    out[:] = unique
    return out


def _same_label_set(left: Sequence | np.ndarray, right: Sequence | np.ndarray) -> bool:
    left_values = np.asarray(left, dtype=object).reshape(-1)
    right_values = np.asarray(right, dtype=object).reshape(-1)
    if left_values.size != right_values.size:
        return False
    return all(_contains_label(right_values, value) for value in left_values) and all(
        _contains_label(left_values, value) for value in right_values
    )


def _contains_label(values: Sequence | np.ndarray, target: object) -> bool:
    return any(_labels_equal(value, target) for value in values)


def _label_mask(labels: Sequence | np.ndarray, target: object) -> np.ndarray:
    return np.asarray([_labels_equal(label, target) for label in np.asarray(labels, dtype=object).reshape(-1)], dtype=bool)


def _count_label(labels: Sequence | np.ndarray, target: object) -> int:
    return int(np.sum(_label_mask(labels, target)))


def _labels_equal(left: object, right: object) -> bool:
    try:
        equal = left == right
    except (TypeError, ValueError):
        return False
    try:
        return bool(equal)
    except (TypeError, ValueError):
        return False


def _common_repetition_count(labels_by_subject: Mapping[Hashable, np.ndarray], classes: np.ndarray, *, requested: int | None) -> int:
    counts = []
    for subject_id, labels in labels_by_subject.items():
        for class_label in classes:
            count = _count_label(labels, class_label)
            if count < 1:
                raise ValueError(f"Subject {subject_id!r} has no samples for class {class_label!r}.")
            counts.append(count)
    available = min(counts)
    if requested is None:
        return available
    if requested > available:
        raise ValueError(f"Requested {requested} repetitions per class, but only {available} are common to all subjects.")
    return int(requested)


def _check_common_alignment_rows(matrices: Mapping[Hashable, np.ndarray]) -> int:
    row_counts = {subject_id: matrix.shape[0] for subject_id, matrix in matrices.items()}
    unique_row_counts = set(row_counts.values())
    if len(unique_row_counts) != 1:
        raise ValueError(f"All subject alignment matrices must have the same row count, got {row_counts}.")
    return int(next(iter(unique_row_counts)))


def _check_subject_keys(features_by_subject, labels_by_subject) -> None:
    if set(features_by_subject) != set(labels_by_subject):
        raise ValueError("features_by_subject and labels_by_subject must have identical subject keys.")
    if len(features_by_subject) < 2:
        raise ValueError("At least two subjects are required.")


def _requested_component_count(n_components: int | float) -> int:
    if n_components == float("inf"):
        return np.iinfo(np.int32).max

    try:
        value = float(n_components)
    except (TypeError, ValueError) as exc:
        raise ValueError("n_components must be a positive integer component count or infinity.") from exc

    if not np.isfinite(value):
        raise ValueError("n_components must be a positive integer component count or infinity.")
    if not value.is_integer():
        raise ValueError(
            "n_components must be an integer component count or infinity; "
            "fractional variance-ratio requests are not supported for M-CCA alignment components."
        )

    requested = int(value)
    if requested < 1:
        raise ValueError("n_components must be a positive integer component count or infinity.")
    return requested


def _normalize_sample_mode(sample_mode: str) -> str:
    normalized = str(sample_mode).strip().lower().replace("-", "_")
    if normalized not in CLASS_ALIGNMENT_SAMPLE_MODES:
        raise ValueError(f"Unknown M-CCA class-alignment sample mode: {sample_mode}. Available modes: {', '.join(CLASS_ALIGNMENT_SAMPLE_MODES)}.")
    return normalized


def _explained_variance_ratio(singular_values: np.ndarray) -> np.ndarray:
    squared = np.asarray(singular_values, dtype=float) ** 2
    total = float(np.sum(squared))
    if total <= 0:
        return np.zeros_like(squared)
    return squared / total


def _feature_matrix(features: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    if matrix.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")
    if matrix.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one column.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values.")
    return matrix


def _object_vector(values: Sequence | np.ndarray) -> np.ndarray:
    """Return a 1D object vector while preserving tuple-like labels.

    ``np.asarray([(1, 2), (3, 4)], dtype=object)`` still creates a 2D object
    array because the tuples have a uniform length.  Alignment anchors may be
    composite metadata keys such as ``(run, stimulus_id)``; flattening those keys
    changes the row count and can make valid M-CCA anchor labels fail the length
    check.  Build the object vector via assignment from ``list(values)`` for
    generic Python sequences so each item remains one scalar label.
    """

    if isinstance(values, np.ndarray):
        if values.ndim == 1:
            return values.astype(object, copy=False).reshape(-1)
        return values.reshape(-1).astype(object)
    try:
        items = list(values)
    except TypeError:
        items = [values]
    vector = np.empty(len(items), dtype=object)
    vector[:] = items
    return vector


def _label_vector(labels: Sequence | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    vector = _object_vector(labels)
    if len(vector) != expected_length:
        raise ValueError(f"{name} length must match feature rows: {len(vector)} != {expected_length}.")
    return vector
