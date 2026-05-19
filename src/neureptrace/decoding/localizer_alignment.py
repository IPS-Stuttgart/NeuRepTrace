"""Independent-localizer alignment utilities for cross-subject decoding.

This module covers the dataset-independent part of workflows that use one data
split to estimate subject alignment and a separate data split for scored
decoding.  Dataset-specific projects provide feature matrices and labels for the
localizer/calibration split, then apply the fitted transforms to their own task
features before model fitting or scoring.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
import hashlib

import numpy as np

LOCALIZER_INITIAL_TEMPLATES = ("first", "mean")


@dataclass(frozen=True)
class ProcrustesTransform:
    """Rigid feature-space map from one aligned matrix into another."""

    source_center: np.ndarray
    target_center: np.ndarray
    rotation: np.ndarray

    def transform(
        self,
        features: Sequence[Sequence[float]] | np.ndarray,
        *,
        block_size: int | None = None,
    ) -> np.ndarray:
        """Apply the transform to rows or row-wise repeated feature blocks."""

        return apply_procrustes_transform(features, self, block_size=block_size)


@dataclass(frozen=True)
class SourceProcrustesTemplate:
    """Source-subject template and one transform per fitted source subject."""

    subject_ids: tuple[Hashable, ...]
    template: np.ndarray
    transforms: Mapping[Hashable, ProcrustesTransform]
    n_iterations: int
    initial_template: str

    def transform(self, subject_id: Hashable, features: Sequence[Sequence[float]] | np.ndarray, *, block_size: int | None = None) -> np.ndarray:
        """Transform rows from one fitted source subject into the source template."""

        try:
            transform = self.transforms[subject_id]
        except KeyError as exc:
            fitted = ", ".join(str(value) for value in self.subject_ids)
            raise KeyError(f"Unknown source subject {subject_id!r}. Fitted subjects: {fitted}.") from exc
        return transform.transform(features, block_size=block_size)


@dataclass(frozen=True)
class SourceLocalizerProcrustesModel:
    """Procrustes alignment fitted from source subjects' localizer data."""

    source_subject_ids: tuple[Hashable, ...]
    classes: np.ndarray
    template: np.ndarray
    transforms: Mapping[Hashable, ProcrustesTransform]
    n_iterations: int
    initial_template: str
    block_size: int | None

    def transform_source(
        self,
        subject_id: Hashable,
        features: Sequence[Sequence[float]] | np.ndarray,
        *,
        block_size: int | None = None,
    ) -> np.ndarray:
        """Transform rows from a fitted source subject into localizer-template space."""

        try:
            transform = self.transforms[subject_id]
        except KeyError as exc:
            fitted = ", ".join(str(value) for value in self.source_subject_ids)
            raise KeyError(f"Unknown source subject {subject_id!r}. Fitted subjects: {fitted}.") from exc
        return transform.transform(features, block_size=self.block_size if block_size is None else block_size)

    def fit_target(
        self,
        features: Sequence[Sequence[float]] | np.ndarray,
        labels: Sequence | np.ndarray,
        *,
        subject_id: Hashable = "target",
        block_size: int | None = None,
        label_permutation_seed: int | str | None = None,
        label_permutation_context: Sequence[object] = (),
    ) -> TargetLocalizerProcrustesTransform:
        """Fit a held-out target transform from target localizer rows."""

        return fit_target_localizer_procrustes(
            self,
            features,
            labels,
            subject_id=subject_id,
            block_size=self.block_size if block_size is None else block_size,
            label_permutation_seed=label_permutation_seed,
            label_permutation_context=label_permutation_context,
        )


@dataclass(frozen=True)
class TargetLocalizerProcrustesTransform:
    """Held-out subject transform fitted only from independent localizer data."""

    subject_id: Hashable
    classes: np.ndarray
    transform: ProcrustesTransform
    block_size: int | None
    label_permutation_seed: int | str | None = None

    def transform_features(
        self,
        features: Sequence[Sequence[float]] | np.ndarray,
        *,
        block_size: int | None = None,
    ) -> np.ndarray:
        """Apply the target-localizer transform to scored task features."""

        return self.transform.transform(features, block_size=self.block_size if block_size is None else block_size)


def fit_procrustes_transform(
    source: Sequence[Sequence[float]] | np.ndarray,
    target: Sequence[Sequence[float]] | np.ndarray,
) -> ProcrustesTransform:
    """Fit an orthogonal Procrustes map from ``source`` rows to ``target`` rows."""

    source_matrix = _feature_matrix(source, name="source")
    target_matrix = _feature_matrix(target, name="target")
    if source_matrix.shape != target_matrix.shape:
        raise ValueError(f"source and target must have the same shape: {source_matrix.shape} != {target_matrix.shape}.")
    if source_matrix.shape[0] < 2:
        raise ValueError("Procrustes alignment requires at least two aligned rows.")

    source_center = np.mean(source_matrix, axis=0)
    target_center = np.mean(target_matrix, axis=0)
    source_centered = source_matrix - source_center
    target_centered = target_matrix - target_center
    cross_covariance = source_centered.T @ target_centered
    left, _singular_values, right_t = np.linalg.svd(cross_covariance, full_matrices=False)
    return ProcrustesTransform(
        source_center=source_center,
        target_center=target_center,
        rotation=left @ right_t,
    )


def apply_procrustes_transform(
    features: Sequence[Sequence[float]] | np.ndarray,
    transform: ProcrustesTransform,
    *,
    block_size: int | None = None,
) -> np.ndarray:
    """Apply a Procrustes transform to feature rows.

    When ``block_size`` is supplied, each row is interpreted as repeated feature
    blocks with that width.  The transform is then applied along the last axis
    and the original row-wise flattened layout is restored.  This supports MEG
    features such as ``channel x time-block`` windows without making the function
    depend on a specific sensor format.
    """

    matrix = _feature_matrix(features, name="features")
    rotation = _feature_matrix(transform.rotation, name="transform.rotation")
    source_center = _center_vector(transform.source_center, expected_width=rotation.shape[0], name="transform.source_center")
    target_center = _center_vector(transform.target_center, expected_width=rotation.shape[1], name="transform.target_center")
    normalized_block_size = _normalize_block_size(block_size)

    if normalized_block_size is None:
        if matrix.shape[1] != rotation.shape[0]:
            raise ValueError(f"features column count does not match transform: {matrix.shape[1]} != {rotation.shape[0]}.")
        return (matrix - source_center) @ rotation + target_center

    if normalized_block_size != rotation.shape[0]:
        raise ValueError(f"block_size must match transform input width: {normalized_block_size} != {rotation.shape[0]}.")
    if matrix.shape[1] % normalized_block_size:
        raise ValueError(f"features column count must be divisible by block_size: {matrix.shape[1]} % {normalized_block_size} != 0.")
    blocks = matrix.reshape(matrix.shape[0], matrix.shape[1] // normalized_block_size, normalized_block_size)
    aligned = (blocks - source_center) @ rotation + target_center
    return aligned.reshape(matrix.shape[0], -1)


def fit_source_only_procrustes_template(
    aligned_by_subject: Mapping[Hashable, Sequence[Sequence[float]] | np.ndarray],
    *,
    n_iterations: int = 3,
    initial_template: str = "first",
) -> SourceProcrustesTemplate:
    """Fit a source-only Procrustes template from row-aligned subject matrices."""

    if not aligned_by_subject:
        raise ValueError("At least one source subject is required.")
    iterations = _normalize_n_iterations(n_iterations)
    initialization = _normalize_initial_template(initial_template)
    subject_ids = tuple(aligned_by_subject.keys())
    matrices = {subject_id: _feature_matrix(matrix, name=f"aligned_by_subject[{subject_id!r}]") for subject_id, matrix in aligned_by_subject.items()}
    _check_common_pattern_shape(matrices)

    if initialization == "first":
        template = np.array(matrices[subject_ids[0]], dtype=float, copy=True)
    else:
        template = np.mean(np.stack([matrices[subject_id] for subject_id in subject_ids], axis=0), axis=0)

    transforms: dict[Hashable, ProcrustesTransform] = {}
    for _ in range(iterations):
        transforms = {subject_id: fit_procrustes_transform(matrices[subject_id], template) for subject_id in subject_ids}
        aligned = [transforms[subject_id].transform(matrices[subject_id]) for subject_id in subject_ids]
        template = np.mean(np.stack(aligned, axis=0), axis=0)
    transforms = {subject_id: fit_procrustes_transform(matrices[subject_id], template) for subject_id in subject_ids}
    return SourceProcrustesTemplate(
        subject_ids=subject_ids,
        template=template,
        transforms=transforms,
        n_iterations=iterations,
        initial_template=initialization,
    )


def fit_source_localizer_procrustes(
    features_by_subject: Mapping[Hashable, Sequence[Sequence[float]] | np.ndarray],
    labels_by_subject: Mapping[Hashable, Sequence | np.ndarray],
    *,
    classes: Sequence | np.ndarray | None = None,
    block_size: int | None = None,
    n_iterations: int = 3,
    initial_template: str = "first",
) -> SourceLocalizerProcrustesModel:
    """Fit source-subject alignment from labeled localizer/calibration rows."""

    _check_subject_keys(features_by_subject, labels_by_subject)
    normalized_block_size = _normalize_block_size(block_size)
    class_order = common_label_values(labels_by_subject) if classes is None else _class_vector(classes)
    if class_order.size < 2:
        raise ValueError("Localizer Procrustes alignment requires at least two common classes.")
    patterns = {
        subject_id: class_pattern_matrix(
            features_by_subject[subject_id],
            labels_by_subject[subject_id],
            classes=class_order,
            block_size=normalized_block_size,
        )
        for subject_id in features_by_subject
    }
    template = fit_source_only_procrustes_template(
        patterns,
        n_iterations=n_iterations,
        initial_template=initial_template,
    )
    return SourceLocalizerProcrustesModel(
        source_subject_ids=template.subject_ids,
        classes=np.array(class_order, copy=True),
        template=template.template,
        transforms=template.transforms,
        n_iterations=template.n_iterations,
        initial_template=template.initial_template,
        block_size=normalized_block_size,
    )


def fit_target_localizer_procrustes(
    source_model: SourceLocalizerProcrustesModel,
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence | np.ndarray,
    *,
    subject_id: Hashable = "target",
    block_size: int | None = None,
    label_permutation_seed: int | str | None = None,
    label_permutation_context: Sequence[object] = (),
) -> TargetLocalizerProcrustesTransform:
    """Fit a held-out target transform into an existing source-localizer template."""

    normalized_block_size = source_model.block_size if block_size is None else _normalize_block_size(block_size)
    target_labels = _label_vector(labels, expected_length=_feature_matrix(features, name="features").shape[0], name="labels")
    if label_permutation_seed is not None:
        target_labels = permuted_labels(
            target_labels,
            seed=label_permutation_seed,
            context=label_permutation_context,
        )
    target_patterns = class_pattern_matrix(
        features,
        target_labels,
        classes=source_model.classes,
        block_size=normalized_block_size,
    )
    return TargetLocalizerProcrustesTransform(
        subject_id=subject_id,
        classes=np.array(source_model.classes, copy=True),
        transform=fit_procrustes_transform(target_patterns, source_model.template),
        block_size=normalized_block_size,
        label_permutation_seed=label_permutation_seed,
    )


def class_pattern_matrix(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence | np.ndarray,
    *,
    classes: Sequence | np.ndarray,
    block_size: int | None = None,
) -> np.ndarray:
    """Build one class-average row per requested class.

    With ``block_size=None`` each class row is the mean feature vector.  With a
    block size, feature rows are reshaped into repeated blocks and each class row
    is averaged over both trials and blocks, yielding one prototype per class in
    block space.
    """

    matrix = _feature_matrix(features, name="features")
    vector = _label_vector(labels, expected_length=matrix.shape[0], name="labels")
    class_order = _class_vector(classes)
    _check_requested_classes(vector, class_order)
    normalized_block_size = _normalize_block_size(block_size)

    if normalized_block_size is None:
        return np.vstack([np.mean(matrix[vector == class_label], axis=0) for class_label in class_order])
    if matrix.shape[1] % normalized_block_size:
        raise ValueError(f"features column count must be divisible by block_size: {matrix.shape[1]} % {normalized_block_size} != 0.")
    blocks = matrix.reshape(matrix.shape[0], matrix.shape[1] // normalized_block_size, normalized_block_size)
    return np.vstack([np.mean(blocks[vector == class_label], axis=(0, 1)) for class_label in class_order])


def common_label_values(labels_by_subject: Mapping[Hashable, Sequence | np.ndarray]) -> np.ndarray:
    """Return labels present in every subject, preserving the first subject's order."""

    if not labels_by_subject:
        return np.array([], dtype=int)
    subject_ids = tuple(labels_by_subject.keys())
    vectors = {subject_id: np.asarray(labels_by_subject[subject_id]).ravel() for subject_id in subject_ids}
    first_values = _unique_in_order(vectors[subject_ids[0]])
    common = [value for value in first_values if all(np.any(vectors[subject_id] == value) for subject_id in subject_ids[1:])]
    return np.asarray(common, dtype=first_values.dtype)


def permuted_labels(labels: Sequence | np.ndarray, *, seed: int | str, context: Sequence[object] = ()) -> np.ndarray:
    """Return a deterministic permutation of labels for calibration controls."""

    vector = np.asarray(labels).ravel()
    seed_values = [_seed_value(seed), *[_seed_value(value) for value in context]]
    rng = np.random.default_rng(np.random.SeedSequence(seed_values))
    return rng.permutation(vector)


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


def _label_vector(labels: Sequence | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    vector = np.asarray(labels).ravel()
    if len(vector) != expected_length:
        raise ValueError(f"{name} length must match feature rows: {len(vector)} != {expected_length}.")
    return vector


def _center_vector(center: Sequence[float] | np.ndarray, *, expected_width: int, name: str) -> np.ndarray:
    vector = np.asarray(center, dtype=float).ravel()
    if vector.shape[0] != expected_width:
        raise ValueError(f"{name} length must match transform width: {vector.shape[0]} != {expected_width}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains non-finite values.")
    return vector


def _class_vector(classes: Sequence | np.ndarray) -> np.ndarray:
    vector = np.asarray(classes).ravel()
    if vector.size == 0:
        raise ValueError("classes must contain at least one label.")
    return vector


def _check_requested_classes(labels: np.ndarray, classes: np.ndarray) -> None:
    missing = [label for label in classes if not np.any(labels == label)]
    if missing:
        raise ValueError(f"classes include labels absent from labels: {missing!r}.")


def _check_subject_keys(features_by_subject, labels_by_subject) -> None:
    if set(features_by_subject) != set(labels_by_subject):
        raise ValueError("features_by_subject and labels_by_subject must have identical subject keys.")
    if not features_by_subject:
        raise ValueError("At least one source subject is required.")


def _check_common_pattern_shape(matrices: Mapping[Hashable, np.ndarray]) -> None:
    shapes = {subject_id: matrix.shape for subject_id, matrix in matrices.items()}
    unique_shapes = set(shapes.values())
    if len(unique_shapes) != 1:
        raise ValueError(f"All subject alignment matrices must have the same shape, got {shapes}.")
    n_rows, _n_features = next(iter(unique_shapes))
    if n_rows < 2:
        raise ValueError("Procrustes alignment requires at least two aligned rows.")


def _normalize_block_size(block_size: int | None) -> int | None:
    if block_size is None:
        return None
    value = int(block_size)
    if value < 1:
        raise ValueError("block_size must be positive or None.")
    return value


def _normalize_n_iterations(n_iterations: int) -> int:
    value = int(n_iterations)
    if value < 0:
        raise ValueError("n_iterations must be non-negative.")
    return value


def _normalize_initial_template(initial_template: str) -> str:
    normalized = str(initial_template).strip().lower().replace("-", "_")
    if normalized not in LOCALIZER_INITIAL_TEMPLATES:
        raise ValueError(f"Unsupported localizer initial_template '{initial_template}'. Available modes: {', '.join(LOCALIZER_INITIAL_TEMPLATES)}.")
    return normalized


def _unique_in_order(values: np.ndarray) -> np.ndarray:
    unique = []
    for value in values:
        if not any(value == existing for existing in unique):
            unique.append(value)
    return np.asarray(unique, dtype=values.dtype)


def _seed_value(value: object) -> int:
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


__all__ = [
    "LOCALIZER_INITIAL_TEMPLATES",
    "ProcrustesTransform",
    "SourceLocalizerProcrustesModel",
    "SourceProcrustesTemplate",
    "TargetLocalizerProcrustesTransform",
    "apply_procrustes_transform",
    "class_pattern_matrix",
    "common_label_values",
    "fit_procrustes_transform",
    "fit_source_localizer_procrustes",
    "fit_source_only_procrustes_template",
    "fit_target_localizer_procrustes",
    "permuted_labels",
]
