"""Dataset-agnostic shared-space LOSO evaluation helpers.

This module lifts the fold mechanics used by project-specific cross-subject
pipelines into NeuRepTrace.  Callers provide already-windowed feature matrices
and labels; dataset packages remain responsible for file loading, event/window
selection, channel adapters, and CSV/reporting conventions.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from neureptrace.decoding.hyperalignment import fit_projection_to_hyperalignment, transform_with_projection
from neureptrace.decoding.hyperalignment_initialization import fit_class_hyperalignment
from neureptrace.decoding.mcca import fit_class_mcca
from neureptrace.decoding.mcca_target import class_alignment_matrix, fit_target_mcca_projection
from neureptrace.decoding.sampling import DEFAULT_CLASS_LIMIT_SEED, DEFAULT_CLASS_LIMIT_SELECTION

SharedSpaceMethod = Literal["hyperalignment", "mcca"]
TargetTransform = Literal["group_mean", "target_unsupervised", "target_labeled"]


@dataclass(frozen=True)
class SubjectFeatureSet:
    """Decoding and optional calibration features for one subject.

    ``features`` and ``labels`` are the rows scored by the LOSO decoder.  When
    ``alignment_features``/``alignment_labels`` are supplied, they are used only
    to fit subject-specific alignment projections.  This covers independent
    localizer/cue calibration data without baking any dataset convention into
    NeuRepTrace.
    """

    subject_id: Hashable
    features: Sequence[Sequence[float]] | np.ndarray
    labels: Sequence | np.ndarray
    alignment_features: Sequence[Sequence[float]] | np.ndarray | None = None
    alignment_labels: Sequence | np.ndarray | None = None


@dataclass(frozen=True)
class SharedSpaceConfig:
    """Configuration for generic shared-space LOSO folds."""

    method: SharedSpaceMethod = "hyperalignment"
    target_transform: TargetTransform = "target_unsupervised"
    sample_mode: str = "class_repetition"
    n_repetitions_per_class: int | None = None
    repetition_selection: str = DEFAULT_CLASS_LIMIT_SELECTION
    repetition_seed: int | str | None = DEFAULT_CLASS_LIMIT_SEED
    n_components: int | float = 64
    hyperalignment_iterations: int = 10
    hyperalignment_template_tolerance: float = 1e-8
    hyperalignment_initialization: str = "pca"
    mcca_regularization: float = 1e-6
    mcca_subject_pca_components: int | float | None = None
    mcca_rank_tolerance: float = 1e-10
    mcca_normalize_components: bool = True
    mcca_target_regularization: float | None = None


@dataclass(frozen=True)
class SharedSpaceFold:
    """Transformed training/test matrices for one LOSO outer fold."""

    test_subject: Hashable
    train_subjects: tuple[Hashable, ...]
    method: str
    target_transform: str
    train_features: np.ndarray
    train_labels: np.ndarray
    test_features: np.ndarray
    test_labels: np.ndarray
    alignment_classes: np.ndarray
    n_components: int
    n_alignment_rows: int
    n_target_alignment_rows: int
    model: Any
    alignment: Any


@dataclass(frozen=True)
class SharedSpaceFoldScore:
    """Prediction metrics for one shared-space LOSO outer fold."""

    test_subject: Hashable
    train_subjects: tuple[Hashable, ...]
    method: str
    target_transform: str
    n_train_trials: int
    n_test_trials: int
    n_components: int
    accuracy: float
    balanced_accuracy: float
    true_labels: np.ndarray
    predicted_labels: np.ndarray
    fold: SharedSpaceFold


def make_loso_shared_space_folds(
    subjects: Sequence[SubjectFeatureSet],
    *,
    config: SharedSpaceConfig | None = None,
    outer_subjects: Sequence[Hashable] | None = None,
    label_shuffle_seed: int | None = None,
) -> list[SharedSpaceFold]:
    """Fit every requested LOSO shared-space fold and return transformed data.

    ``label_shuffle_seed`` shuffles training labels independently within each
    outer fold and training subject.  Validation/test labels are never shuffled.
    When a subject has explicit ``alignment_labels``, those labels are left
    untouched so independent calibration/localizer data can remain aligned while
    the scored decoder labels are shuffled.
    """

    by_subject = _subject_mapping(subjects)
    test_subjects = tuple(by_subject) if outer_subjects is None else tuple(outer_subjects)
    unknown = [subject_id for subject_id in test_subjects if subject_id not in by_subject]
    if unknown:
        raise ValueError(f"outer_subjects contains unknown subject ids: {unknown!r}.")
    return [fit_loso_shared_space_fold(by_subject, test_subject, config=config, label_shuffle_seed=label_shuffle_seed) for test_subject in test_subjects]


def fit_loso_shared_space_fold(
    subjects: Sequence[SubjectFeatureSet] | Mapping[Hashable, SubjectFeatureSet],
    test_subject: Hashable,
    *,
    config: SharedSpaceConfig | None = None,
    label_shuffle_seed: int | None = None,
) -> SharedSpaceFold:
    """Fit one dataset-agnostic LOSO shared-space fold.

    The returned matrices are ready for a downstream classifier.  This function
    assumes decoding and alignment features have the same column layout.  Projects
    with separate sensor/time adapters can still reuse the lower-level fitted
    ``model`` and ``alignment`` objects stored on the returned fold.
    """

    config = _checked_config(config or SharedSpaceConfig())
    by_subject = _subject_mapping(subjects.values() if isinstance(subjects, Mapping) else subjects)
    if test_subject not in by_subject:
        raise ValueError(f"Unknown test_subject {test_subject!r}.")

    train_ids = tuple(subject_id for subject_id in by_subject if subject_id != test_subject)
    if len(train_ids) < 2:
        raise ValueError("Shared-space LOSO requires at least two training subjects.")

    train_sets = {subject_id: by_subject[subject_id] for subject_id in train_ids}
    test_set = by_subject[test_subject]
    train_decode_features = {subject_id: _decode_features(subject) for subject_id, subject in train_sets.items()}
    train_decode_labels = {
        subject_id: _maybe_shuffle_labels(_decode_labels(subject), label_shuffle_seed, context=(test_subject, subject_id))
        for subject_id, subject in train_sets.items()
    }
    train_alignment_features = {subject_id: _alignment_features(subject) for subject_id, subject in train_sets.items()}
    train_alignment_labels = {
        subject_id: _alignment_labels(subject, fallback=train_decode_labels[subject_id]) for subject_id, subject in train_sets.items()
    }

    if config.method == "hyperalignment":
        model, alignment = fit_class_hyperalignment(
            train_alignment_features,
            train_alignment_labels,
            sample_mode=config.sample_mode,
            n_repetitions_per_class=config.n_repetitions_per_class,
            repetition_selection=config.repetition_selection,
            repetition_seed=config.repetition_seed,
            n_components=config.n_components,
            n_iterations=config.hyperalignment_iterations,
            template_tolerance=config.hyperalignment_template_tolerance,
            initialization=config.hyperalignment_initialization,
        )
        transformed_train = [model.transform(subject_id, train_decode_features[subject_id]) for subject_id in train_ids]
        transformed_test, n_target_rows = _transform_hyperalignment_target(model, alignment, test_set, config)
    else:
        model, alignment = fit_class_mcca(
            train_alignment_features,
            train_alignment_labels,
            sample_mode=config.sample_mode,
            n_repetitions_per_class=config.n_repetitions_per_class,
            repetition_selection=config.repetition_selection,
            repetition_seed=config.repetition_seed,
            n_components=config.n_components,
            regularization=config.mcca_regularization,
            subject_pca_components=config.mcca_subject_pca_components,
            rank_tolerance=config.mcca_rank_tolerance,
            normalize_components=config.mcca_normalize_components,
        )
        transformed_train = [model.transform(subject_id, train_decode_features[subject_id]) for subject_id in train_ids]
        transformed_test, n_target_rows = _transform_mcca_target(model, alignment, test_set, config)

    return SharedSpaceFold(
        test_subject=test_subject,
        train_subjects=train_ids,
        method=config.method,
        target_transform=config.target_transform,
        train_features=np.vstack(transformed_train),
        train_labels=np.concatenate([train_decode_labels[subject_id] for subject_id in train_ids]),
        test_features=transformed_test,
        test_labels=_decode_labels(test_set),
        alignment_classes=np.asarray(alignment.classes),
        n_components=int(model.n_components),
        n_alignment_rows=int(next(iter(alignment.aligned_by_subject.values())).shape[0]),
        n_target_alignment_rows=int(n_target_rows),
        model=model,
        alignment=alignment,
    )


def evaluate_loso_shared_space(
    subjects: Sequence[SubjectFeatureSet],
    *,
    fit_estimator: Callable[[np.ndarray, np.ndarray], Any],
    predict: Callable[[Any, np.ndarray], Sequence] | None = None,
    config: SharedSpaceConfig | None = None,
    outer_subjects: Sequence[Hashable] | None = None,
    label_shuffle_seed: int | None = None,
) -> list[SharedSpaceFoldScore]:
    """Run LOSO shared-space folds and score an arbitrary classifier."""

    scores: list[SharedSpaceFoldScore] = []
    for fold in make_loso_shared_space_folds(subjects, config=config, outer_subjects=outer_subjects, label_shuffle_seed=label_shuffle_seed):
        estimator = fit_estimator(fold.train_features, fold.train_labels)
        predicted = _predict(estimator, fold.test_features, predict)
        scores.append(
            SharedSpaceFoldScore(
                test_subject=fold.test_subject,
                train_subjects=fold.train_subjects,
                method=fold.method,
                target_transform=fold.target_transform,
                n_train_trials=int(fold.train_features.shape[0]),
                n_test_trials=int(fold.test_features.shape[0]),
                n_components=fold.n_components,
                accuracy=float(accuracy_score(fold.test_labels, predicted)),
                balanced_accuracy=float(balanced_accuracy_score(fold.test_labels, predicted)),
                true_labels=fold.test_labels,
                predicted_labels=predicted,
                fold=fold,
            )
        )
    return scores


def _transform_hyperalignment_target(model, alignment, subject: SubjectFeatureSet, config: SharedSpaceConfig) -> tuple[np.ndarray, int]:
    features = _decode_features(subject)
    if config.target_transform == "group_mean":
        return model.transform_group(features), 0
    if config.target_transform == "target_unsupervised":
        return model.transform_group(features, feature_mean=np.mean(features, axis=0)), 0
    target_aligned = _target_alignment_matrix(subject, alignment, config)
    projection = fit_projection_to_hyperalignment(target_aligned, template=model.template)
    return transform_with_projection(features, projection), int(target_aligned.shape[0])


def _transform_mcca_target(model, alignment, subject: SubjectFeatureSet, config: SharedSpaceConfig) -> tuple[np.ndarray, int]:
    features = _decode_features(subject)
    if config.target_transform == "group_mean":
        return model.transform_group(features), 0
    if config.target_transform == "target_unsupervised":
        return model.transform_group(features, feature_mean=np.mean(features, axis=0)), 0
    target_aligned = _target_alignment_matrix(subject, alignment, config)
    projection = fit_target_mcca_projection(target_aligned, model, regularization=config.mcca_target_regularization)
    return projection.transform(features), int(target_aligned.shape[0])


def _target_alignment_matrix(subject: SubjectFeatureSet, alignment: Any, config: SharedSpaceConfig) -> np.ndarray:
    if subject.alignment_features is None or subject.alignment_labels is None:
        raise ValueError(
            "target_labeled requires alignment_features and alignment_labels for the held-out subject; "
            "do not use scored test labels implicitly for target projection."
        )
    return class_alignment_matrix(
        subject.alignment_features,
        subject.alignment_labels,
        classes=alignment.classes,
        sample_mode=alignment.sample_mode,
        n_repetitions_per_class=alignment.n_repetitions_per_class,
        repetition_selection=alignment.repetition_selection or config.repetition_selection,
        repetition_seed=alignment.repetition_seed if alignment.repetition_seed is not None else config.repetition_seed,
    )


def _decode_features(subject: SubjectFeatureSet) -> np.ndarray:
    return _feature_matrix(subject.features, name=f"features[{subject.subject_id!r}]")


def _decode_labels(subject: SubjectFeatureSet) -> np.ndarray:
    return _label_vector(subject.labels, expected_length=_decode_features(subject).shape[0], name=f"labels[{subject.subject_id!r}]")


def _alignment_features(subject: SubjectFeatureSet) -> np.ndarray:
    if subject.alignment_features is None:
        return _decode_features(subject)
    return _feature_matrix(subject.alignment_features, name=f"alignment_features[{subject.subject_id!r}]")


def _alignment_labels(subject: SubjectFeatureSet, *, fallback: np.ndarray) -> np.ndarray:
    if subject.alignment_labels is None:
        if subject.alignment_features is not None and subject.alignment_features is not subject.features:
            raise ValueError(f"alignment_labels are required when alignment_features are provided for subject {subject.subject_id!r}.")
        return fallback
    return _label_vector(subject.alignment_labels, expected_length=_alignment_features(subject).shape[0], name=f"alignment_labels[{subject.subject_id!r}]")


def _subject_mapping(subjects: Sequence[SubjectFeatureSet]) -> dict[Hashable, SubjectFeatureSet]:
    mapping: dict[Hashable, SubjectFeatureSet] = {}
    for subject in subjects:
        if subject.subject_id in mapping:
            raise ValueError(f"Duplicate subject_id {subject.subject_id!r}.")
        mapping[subject.subject_id] = subject
    if len(mapping) < 3:
        raise ValueError("Shared-space LOSO requires at least three subjects.")
    return mapping


def _feature_matrix(features: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must have at least one row and one column.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values.")
    return matrix


def _label_vector(labels: Sequence | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    vector = np.asarray(labels).ravel()
    if vector.shape[0] != expected_length:
        raise ValueError(f"{name} length must match feature rows: {vector.shape[0]} != {expected_length}.")
    return vector


def _maybe_shuffle_labels(labels: np.ndarray, seed: int | None, *, context: tuple[Hashable, ...]) -> np.ndarray:
    if seed is None:
        return labels
    seed_values = [int(seed), *[_stable_seed_value(value) for value in context]]
    rng = np.random.default_rng(np.random.SeedSequence(seed_values))
    return rng.permutation(labels)


def _stable_seed_value(value: Hashable) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    return sum((index + 1) * ord(char) for index, char in enumerate(str(value)))


def _predict(estimator: Any, features: np.ndarray, predict: Callable[[Any, np.ndarray], Sequence] | None) -> np.ndarray:
    if predict is not None:
        return np.asarray(predict(estimator, features))
    if not hasattr(estimator, "predict"):
        raise ValueError("estimator must provide predict(features), or pass a predict callback.")
    return np.asarray(estimator.predict(features))



def _normalize_method(method: str) -> str:
    normalized = str(method).strip().lower().replace("-", "_")
    aliases = {
        "m_cca": "mcca",
        "multiset_cca": "mcca",
        "procrustes": "hyperalignment",
        "procrustes_hyperalignment": "hyperalignment",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"hyperalignment", "mcca"}:
        raise ValueError("method must be 'hyperalignment' or 'mcca'.")
    return normalized


def _normalize_target_transform(target_transform: str) -> str:
    normalized = str(target_transform).strip().lower().replace("-", "_")
    aliases = {
        "group": "group_mean",
        "group_average": "group_mean",
        "group_projection": "group_mean",
        "target_calibrated": "target_labeled",
        "cue_target_calibrated": "target_labeled",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"group_mean", "target_unsupervised", "target_labeled"}:
        raise ValueError("target_transform must be 'group_mean', 'target_unsupervised', or 'target_labeled'.")
    return normalized


def _checked_config(config: SharedSpaceConfig) -> SharedSpaceConfig:
    method = _normalize_method(config.method)
    target = _normalize_target_transform(config.target_transform)
    if config.n_repetitions_per_class is not None and int(config.n_repetitions_per_class) < 1:
        raise ValueError("n_repetitions_per_class must be positive or None.")
    if float(config.mcca_regularization) < 0:
        raise ValueError("mcca_regularization must be non-negative.")
    if config.mcca_target_regularization is not None and float(config.mcca_target_regularization) < 0:
        raise ValueError("mcca_target_regularization must be non-negative or None.")
    if method == config.method and target == config.target_transform:
        return config
    return SharedSpaceConfig(
        method=method,  # type: ignore[arg-type]
        target_transform=target,  # type: ignore[arg-type]
        sample_mode=config.sample_mode,
        n_repetitions_per_class=config.n_repetitions_per_class,
        repetition_selection=config.repetition_selection,
        repetition_seed=config.repetition_seed,
        n_components=config.n_components,
        hyperalignment_iterations=config.hyperalignment_iterations,
        hyperalignment_template_tolerance=config.hyperalignment_template_tolerance,
        hyperalignment_initialization=config.hyperalignment_initialization,
        mcca_regularization=config.mcca_regularization,
        mcca_subject_pca_components=config.mcca_subject_pca_components,
        mcca_rank_tolerance=config.mcca_rank_tolerance,
        mcca_normalize_components=config.mcca_normalize_components,
        mcca_target_regularization=config.mcca_target_regularization,
    )


__all__ = [
    "SharedSpaceConfig",
    "SharedSpaceFold",
    "SharedSpaceFoldScore",
    "SubjectFeatureSet",
    "evaluate_loso_shared_space",
    "fit_loso_shared_space_fold",
    "make_loso_shared_space_folds",
]
