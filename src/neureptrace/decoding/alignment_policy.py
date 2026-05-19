"""Policy layer for source-only cross-subject alignment.

This module keeps dataset-specific code out of NeuRepTrace while making the
alignment semantics explicit enough for PyMEGDec-style transfer workflows:

* the common-space model is fitted on source/training subjects only;
* held-out target subjects can be handled by a group projection, by an
  unsupervised target-centering estimate, or by a separate labeled calibration
  / localizer matrix;
* source and target-calibration labels can be shuffled for leakage/control
  analyses; and
* every fitted policy emits compact provenance that downstream result tables can
  write without knowing whether the underlying model is hyperalignment or M-CCA.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from neureptrace.decoding.hyperalignment import (
    ClassAlignment as HyperalignmentClassAlignment,
    HyperalignmentModel,
    SubjectHyperalignmentProjection,
    fit_class_hyperalignment,
    fit_projection_to_hyperalignment,
    transform_with_projection,
)
from neureptrace.decoding.mcca import ClassAlignment as MCCAClassAlignment
from neureptrace.decoding.mcca import MCCAModel, fit_class_mcca
from neureptrace.decoding.mcca_target import TargetMCCAProjection, class_alignment_matrix, fit_target_mcca_projection
from neureptrace.decoding.sampling import DEFAULT_CLASS_LIMIT_SEED, DEFAULT_CLASS_LIMIT_SELECTION

ALIGNMENT_METHODS = ("none", "hyperalignment", "mcca")
TARGET_POLICIES = ("none", "group", "unsupervised_centering", "calibration")
LABEL_POLICIES = ("normal", "shuffle")
FIT_DATA_POLICIES = ("source_only",)


@dataclass(frozen=True)
class AlignmentPlan:
    """Declarative alignment policy for cross-subject transfer.

    Parameters
    ----------
    method:
        ``"none"``, ``"hyperalignment"`` / ``"procrustes"``, or ``"mcca"``.
    fit_data:
        Currently only ``"source_only"`` is supported.  Target calibration rows
        are used only to fit a target projection after the source model exists.
    target_policy:
        How a held-out target subject should be transformed:

        ``"none"``
            Do not transform target rows.
        ``"group"``
            Use the training-subject average projection and average training
            feature mean.
        ``"unsupervised_centering"``
            Use the training-subject average projection but center with an
            unlabeled target matrix supplied as ``target_centering_features``.
        ``"calibration"``
            Fit a target-subject projection from labeled calibration/localizer
            rows supplied as ``target_features`` and ``target_labels``.
    source_label_policy:
        ``"normal"`` or ``"shuffle"`` for source labels used to build alignment
        anchors.
    target_label_policy:
        ``"normal"`` or ``"shuffle"`` for labeled target-calibration anchors.
        It is ignored unless ``target_policy="calibration"``.
    random_state:
        Seed used by source/target label-shuffle controls.  Set to ``None`` for
        a fresh non-deterministic shuffle.
    sample_mode, n_repetitions_per_class, repetition_selection, repetition_seed:
        Anchor construction options passed to the existing class-alignment
        helpers.
    n_components, n_iterations, template_tolerance:
        Hyperalignment / common-space parameters.
    regularization, subject_pca_components, rank_tolerance, normalize_components:
        M-CCA-specific parameters.
    """

    method: str = "none"
    fit_data: str = "source_only"
    target_policy: str = "none"
    source_label_policy: str = "normal"
    target_label_policy: str = "normal"
    random_state: int | None = 0
    sample_mode: str = "class_mean"
    n_repetitions_per_class: int | None = None
    repetition_selection: str = DEFAULT_CLASS_LIMIT_SELECTION
    repetition_seed: int | str | None = DEFAULT_CLASS_LIMIT_SEED
    n_components: int | float = 64
    n_iterations: int = 10
    template_tolerance: float = 1e-8
    regularization: float = 1e-6
    subject_pca_components: int | float | None = None
    rank_tolerance: float = 1e-10
    normalize_components: bool = True


AlignmentModel = HyperalignmentModel | MCCAModel
AlignmentClassRows = HyperalignmentClassAlignment | MCCAClassAlignment
TargetProjection = SubjectHyperalignmentProjection | TargetMCCAProjection


@dataclass(frozen=True)
class FittedAlignmentPolicy:
    """Fitted alignment policy plus provenance and transformation helpers."""

    plan: AlignmentPlan
    model: AlignmentModel | None
    source_alignment: AlignmentClassRows | None
    target_projection: TargetProjection | None
    target_feature_mean: np.ndarray | None
    provenance: Mapping[str, Any]

    def transform_source(self, subject_id: Hashable, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        """Transform rows from a fitted source subject."""

        matrix = _feature_matrix(features, name="features")
        if self.plan.method == "none":
            return matrix
        if self.model is None:
            raise RuntimeError("Alignment model is unavailable.")
        return self.model.transform(subject_id, matrix)

    def transform_target(
        self,
        features: Sequence[Sequence[float]] | np.ndarray,
        *,
        feature_mean: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Transform rows from a held-out target subject according to the policy."""

        matrix = _feature_matrix(features, name="features")
        if self.plan.method == "none" or self.plan.target_policy == "none":
            return matrix
        if self.model is None:
            raise RuntimeError("Alignment model is unavailable.")

        if self.plan.target_policy == "group":
            return self.model.transform_group(matrix)

        if self.plan.target_policy == "unsupervised_centering":
            mean = self.target_feature_mean if feature_mean is None else np.asarray(feature_mean, dtype=float).ravel()
            if mean is None:
                raise ValueError("feature_mean or target_centering_features is required for unsupervised target centering.")
            return self.model.transform_group(matrix, feature_mean=mean)

        if self.plan.target_policy == "calibration":
            if self.target_projection is None:
                raise RuntimeError("Target calibration projection is unavailable.")
            if isinstance(self.model, HyperalignmentModel):
                if not isinstance(self.target_projection, SubjectHyperalignmentProjection):
                    raise TypeError("Expected a hyperalignment target projection.")
                return transform_with_projection(matrix, self.target_projection)
            if isinstance(self.model, MCCAModel):
                if not isinstance(self.target_projection, TargetMCCAProjection):
                    raise TypeError("Expected an M-CCA target projection.")
                return self.target_projection.transform(matrix)

        raise ValueError(f"Unsupported target policy: {self.plan.target_policy!r}.")


def fit_alignment_policy(
    features_by_subject: Mapping[Hashable, Sequence[Sequence[float]] | np.ndarray],
    labels_by_subject: Mapping[Hashable, Sequence | np.ndarray],
    *,
    plan: AlignmentPlan | None = None,
    target_features: Sequence[Sequence[float]] | np.ndarray | None = None,
    target_labels: Sequence | np.ndarray | None = None,
    target_id: Hashable | None = None,
    target_centering_features: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> FittedAlignmentPolicy:
    """Fit a source-only alignment policy.

    ``features_by_subject`` and ``labels_by_subject`` define the source/training
    subjects.  The source model is never fitted with target rows.  If
    ``target_policy="calibration"``, ``target_features`` and ``target_labels``
    are used afterwards to estimate only a held-out target projection.
    """

    normalized_plan = normalize_alignment_plan(plan or AlignmentPlan())
    source_subject_ids = tuple(features_by_subject.keys())
    if normalized_plan.method == "none":
        return FittedAlignmentPolicy(
            plan=normalized_plan,
            model=None,
            source_alignment=None,
            target_projection=None,
            target_feature_mean=None,
            provenance=_provenance(
                normalized_plan,
                source_subject_ids=source_subject_ids,
                source_alignment=None,
                model=None,
                target_id=target_id,
                target_transform="none",
                target_centering="none",
                target_calibration_rows=0,
            ),
        )

    source_labels = _labels_by_subject_after_policy(
        labels_by_subject,
        policy=normalized_plan.source_label_policy,
        random_state=normalized_plan.random_state,
    )
    if normalized_plan.method == "hyperalignment":
        model, source_alignment = fit_class_hyperalignment(
            features_by_subject,
            source_labels,
            sample_mode=normalized_plan.sample_mode,
            n_repetitions_per_class=normalized_plan.n_repetitions_per_class,
            repetition_selection=normalized_plan.repetition_selection,
            repetition_seed=normalized_plan.repetition_seed,
            n_components=normalized_plan.n_components,
            n_iterations=normalized_plan.n_iterations,
            template_tolerance=normalized_plan.template_tolerance,
        )
    elif normalized_plan.method == "mcca":
        model, source_alignment = fit_class_mcca(
            features_by_subject,
            source_labels,
            sample_mode=normalized_plan.sample_mode,
            n_repetitions_per_class=normalized_plan.n_repetitions_per_class,
            repetition_selection=normalized_plan.repetition_selection,
            repetition_seed=normalized_plan.repetition_seed,
            n_components=normalized_plan.n_components,
            regularization=normalized_plan.regularization,
            subject_pca_components=normalized_plan.subject_pca_components,
            rank_tolerance=normalized_plan.rank_tolerance,
            normalize_components=normalized_plan.normalize_components,
        )
    else:
        raise ValueError(f"Unsupported alignment method: {normalized_plan.method!r}.")

    target_projection: TargetProjection | None = None
    target_feature_mean: np.ndarray | None = None
    target_transform = "none"
    target_centering = "none"
    target_calibration_rows = 0

    if normalized_plan.target_policy == "group":
        target_transform = "group_projection"
        target_centering = "training_group_mean"
    elif normalized_plan.target_policy == "unsupervised_centering":
        if target_centering_features is None:
            raise ValueError("target_centering_features is required for target_policy='unsupervised_centering'.")
        centering_matrix = _feature_matrix(target_centering_features, name="target_centering_features")
        target_feature_mean = np.mean(centering_matrix, axis=0)
        target_transform = "group_projection"
        target_centering = "target_unsupervised_mean"
    elif normalized_plan.target_policy == "calibration":
        if target_features is None or target_labels is None:
            raise ValueError("target_features and target_labels are required for target_policy='calibration'.")
        target_matrix = _feature_matrix(target_features, name="target_features")
        target_vector = _target_labels_after_policy(
            target_labels,
            policy=normalized_plan.target_label_policy,
            random_state=normalized_plan.random_state,
        )
        target_aligned = class_alignment_matrix(
            target_matrix,
            target_vector,
            classes=source_alignment.classes,
            sample_mode=source_alignment.sample_mode,
            n_repetitions_per_class=source_alignment.n_repetitions_per_class,
            repetition_selection=source_alignment.repetition_selection or normalized_plan.repetition_selection,
            repetition_seed=source_alignment.repetition_seed if source_alignment.repetition_seed is not None else normalized_plan.repetition_seed,
        )
        if isinstance(model, HyperalignmentModel):
            target_projection = fit_projection_to_hyperalignment(target_aligned, template=model.template)
        else:
            target_projection = fit_target_mcca_projection(target_aligned, model, regularization=normalized_plan.regularization)
        target_transform = "target_calibration_projection"
        target_centering = "target_calibration_mean"
        target_calibration_rows = int(target_aligned.shape[0])

    return FittedAlignmentPolicy(
        plan=normalized_plan,
        model=model,
        source_alignment=source_alignment,
        target_projection=target_projection,
        target_feature_mean=target_feature_mean,
        provenance=_provenance(
            normalized_plan,
            source_subject_ids=source_subject_ids,
            source_alignment=source_alignment,
            model=model,
            target_id=target_id,
            target_transform=target_transform,
            target_centering=target_centering,
            target_calibration_rows=target_calibration_rows,
        ),
    )


def normalize_alignment_plan(plan: AlignmentPlan) -> AlignmentPlan:
    """Return a validated plan with canonical policy names."""

    method = _normalize_alignment_method(plan.method)
    fit_data = _normalize_fit_data(plan.fit_data)
    target_policy = _normalize_target_policy(plan.target_policy)
    source_label_policy = _normalize_label_policy(plan.source_label_policy, name="source_label_policy")
    target_label_policy = _normalize_label_policy(plan.target_label_policy, name="target_label_policy")
    if method == "none" and target_policy != "none":
        raise ValueError("target_policy must be 'none' when method='none'.")
    if plan.n_repetitions_per_class is not None and int(plan.n_repetitions_per_class) < 1:
        raise ValueError("n_repetitions_per_class must be positive or None.")
    if int(plan.n_iterations) < 1:
        raise ValueError("n_iterations must be positive.")
    if float(plan.regularization) < 0:
        raise ValueError("regularization must be non-negative.")
    if float(plan.rank_tolerance) < 0:
        raise ValueError("rank_tolerance must be non-negative.")
    return AlignmentPlan(
        method=method,
        fit_data=fit_data,
        target_policy=target_policy,
        source_label_policy=source_label_policy,
        target_label_policy=target_label_policy,
        random_state=plan.random_state,
        sample_mode=plan.sample_mode,
        n_repetitions_per_class=None if plan.n_repetitions_per_class is None else int(plan.n_repetitions_per_class),
        repetition_selection=plan.repetition_selection,
        repetition_seed=plan.repetition_seed,
        n_components=plan.n_components,
        n_iterations=int(plan.n_iterations),
        template_tolerance=float(plan.template_tolerance),
        regularization=float(plan.regularization),
        subject_pca_components=plan.subject_pca_components,
        rank_tolerance=float(plan.rank_tolerance),
        normalize_components=bool(plan.normalize_components),
    )


def _normalize_alignment_method(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"procrustes", "procrustes_hyperalignment", "train_class_procrustes"}:
        normalized = "hyperalignment"
    if normalized in {"m_cca", "multiset_cca", "multi_set_cca"}:
        normalized = "mcca"
    if normalized not in ALIGNMENT_METHODS:
        raise ValueError(f"Unknown alignment method {value!r}. Available methods: {', '.join(ALIGNMENT_METHODS)}.")
    return normalized


def _normalize_fit_data(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"source", "source_subjects", "train", "train_only", "training_only"}:
        normalized = "source_only"
    if normalized not in FIT_DATA_POLICIES:
        raise ValueError(f"Unknown fit_data policy {value!r}. Available policies: {', '.join(FIT_DATA_POLICIES)}.")
    return normalized


def _normalize_target_policy(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"average", "average_projection", "group_projection", "training_group", "training_group_mean"}:
        normalized = "group"
    if normalized in {"target_centering", "unsupervised", "unsupervised_center", "target_unsupervised", "target_unsupervised_centering"}:
        normalized = "unsupervised_centering"
    if normalized in {"calibrated", "target_calibration", "calibration_projection", "localizer"}:
        normalized = "calibration"
    if normalized not in TARGET_POLICIES:
        raise ValueError(f"Unknown target policy {value!r}. Available policies: {', '.join(TARGET_POLICIES)}.")
    return normalized


def _normalize_label_policy(value: str, *, name: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"", "none", "identity", "normal_labels"}:
        normalized = "normal"
    if normalized in {"shuffled", "label_shuffle", "shuffle_labels"}:
        normalized = "shuffle"
    if normalized not in LABEL_POLICIES:
        raise ValueError(f"{name} must be one of {LABEL_POLICIES}.")
    return normalized


def _labels_by_subject_after_policy(
    labels_by_subject: Mapping[Hashable, Sequence | np.ndarray],
    *,
    policy: str,
    random_state: int | None,
) -> dict[Hashable, np.ndarray]:
    labels = {}
    for subject_position, (subject_id, values) in enumerate(labels_by_subject.items()):
        vector = _label_vector(values, name=f"labels_by_subject[{subject_id!r}]")
        if policy == "shuffle":
            vector = _shuffle_label_vector(vector, random_state=random_state, seed_context=subject_position)
        labels[subject_id] = vector
    return labels


def _target_labels_after_policy(labels: Sequence | np.ndarray, *, policy: str, random_state: int | None) -> np.ndarray:
    vector = _label_vector(labels, name="target_labels")
    if policy == "shuffle":
        return _shuffle_label_vector(vector, random_state=random_state, seed_context=1_000_003)
    return vector


def _shuffle_label_vector(labels: np.ndarray, *, random_state: int | None, seed_context: int) -> np.ndarray:
    if random_state is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(np.random.SeedSequence([int(random_state), int(seed_context)]))
    return labels[rng.permutation(labels.shape[0])]


def _provenance(
    plan: AlignmentPlan,
    *,
    source_subject_ids: tuple[Hashable, ...],
    source_alignment: AlignmentClassRows | None,
    model: AlignmentModel | None,
    target_id: Hashable | None,
    target_transform: str,
    target_centering: str,
    target_calibration_rows: int,
) -> dict[str, Any]:
    classes = () if source_alignment is None else tuple(_python_scalar(value) for value in np.asarray(source_alignment.classes).tolist())
    return {
        "alignment_method": plan.method,
        "alignment_fit_data": plan.fit_data,
        "alignment_target_policy": plan.target_policy,
        "alignment_target_transform": target_transform,
        "alignment_target_centering": target_centering,
        "alignment_source_label_policy": plan.source_label_policy,
        "alignment_target_label_policy": plan.target_label_policy if plan.target_policy == "calibration" else "unused",
        "alignment_random_state": plan.random_state,
        "alignment_source_subjects": source_subject_ids,
        "alignment_target_id": target_id,
        "alignment_classes": classes,
        "alignment_n_classes": len(classes),
        "alignment_sample_mode": None if source_alignment is None else source_alignment.sample_mode,
        "alignment_n_repetitions_per_class": None if source_alignment is None else source_alignment.n_repetitions_per_class,
        "alignment_repetition_selection": None if source_alignment is None else source_alignment.repetition_selection,
        "alignment_repetition_seed": None if source_alignment is None else source_alignment.repetition_seed,
        "alignment_n_components": None if model is None else model.n_components,
        "alignment_target_calibration_rows": int(target_calibration_rows),
    }


def _python_scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def _feature_matrix(features: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must have at least one row and one column.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values.")
    return matrix


def _label_vector(labels: Sequence | np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(labels).ravel()
    if vector.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one value.")
    return vector


__all__ = [
    "ALIGNMENT_METHODS",
    "FIT_DATA_POLICIES",
    "LABEL_POLICIES",
    "TARGET_POLICIES",
    "AlignmentPlan",
    "FittedAlignmentPolicy",
    "fit_alignment_policy",
    "normalize_alignment_plan",
]
