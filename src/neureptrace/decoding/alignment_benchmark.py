"""Generic LOSO benchmarks for cross-subject alignment models.

This module keeps dataset loading, sensor geometry, and file naming outside
NeuRepTrace.  Callers provide already-windowed feature matrices and labels for
one subject at a time; NeuRepTrace fits the requested shared-space alignment,
projects source and held-out subjects, trains a classifier, and returns stable
row dictionaries for CSV/JSON export.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from math import comb
from typing import Any
from zlib import crc32

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from neureptrace.decoding.classifiers import get_default_classifier_param, train_multiclass_classifier
from neureptrace.decoding.hyperalignment import fit_projection_to_hyperalignment
from neureptrace.decoding.hyperalignment_initialization import fit_class_hyperalignment
from neureptrace.decoding.mcca import fit_class_mcca
from neureptrace.decoding.mcca_target import TargetMCCAProjection, class_alignment_matrix, fit_target_mcca_projection
from neureptrace.decoding.windowed import fit_window_model, predict_window_model, transform_window_features

ALIGNMENT_METHODS = ("hyperalignment", "mcca")
TARGET_CALIBRATION_MODES = ("none", "heldout_trials", "alignment_data")
TARGET_CENTERING_MODES = ("group_mean", "target_unsupervised")


@dataclass(frozen=True)
class AlignmentSubjectData:
    """Precomputed feature data for one subject.

    ``features`` and ``labels`` are the rows to decode and score.  Optional
    ``alignment_features``/``alignment_labels`` let a project fit the shared
    space from a separate calibration/localizer matrix while still decoding the
    primary feature matrix.  When omitted, the decoding rows are reused for
    alignment.
    """

    subject_id: Hashable
    features: Sequence[Sequence[float]] | np.ndarray
    labels: Sequence | np.ndarray
    alignment_features: Sequence[Sequence[float]] | np.ndarray | None = None
    alignment_labels: Sequence | np.ndarray | None = None


@dataclass(frozen=True)
class AlignmentBenchmarkConfig:
    """Parameters for generic cross-subject alignment benchmarking."""

    method: str = "hyperalignment"
    sample_mode: str = "class_repetition"
    n_repetitions_per_class: int | None = None
    n_components: int | float = 64
    hyperalignment_iterations: int = 10
    hyperalignment_initialization: str = "pca"
    mcca_regularization: float = 1e-6
    mcca_subject_pca_components: int | float | None = None
    target_calibration_mode: str = "none"
    target_calibration_trials_per_class: int = 0
    target_centering: str = "target_unsupervised"
    target_projection_regularization: float | None = None
    classifier: str = "multiclass-svm"
    classifier_param: Any = None
    components_pca: int | float = float("inf")
    random_state: int | None = 0
    chance_classes: int | None = None
    signflip_permutations: int = 10_000
    signflip_seed: int | None = 0


@dataclass(frozen=True)
class AlignmentOuterFoldResult:
    """Artifacts from one held-out-subject alignment fold."""

    outer_row: dict[str, Any]
    prediction_rows: list[dict[str, Any]]
    alignment_model: Any
    class_alignment: Any
    model_bundle: Any


@dataclass(frozen=True)
class _PreparedSubjectData:
    subject_id: Hashable
    features: np.ndarray
    labels: np.ndarray
    alignment_features: np.ndarray
    alignment_labels: np.ndarray
    alignment_features_explicit: bool
    alignment_labels_explicit: bool
    alignment_uses_decode_labels: bool


ProjectionTransform = Callable[
    [Sequence[Sequence[float]] | np.ndarray],
    np.ndarray,
]


def evaluate_alignment_loso(
    subjects: Iterable[AlignmentSubjectData],
    *,
    config: AlignmentBenchmarkConfig | None = None,
    outer_subjects: Sequence[Hashable] | None = None,
    fit_model: Callable[[np.ndarray, np.ndarray], Any] | None = None,
    projection_transform: Callable[..., np.ndarray] | None = None,
    progress: Callable[[str], None] | None = None,
    label_shuffle_seed: int | None = None,
) -> dict[str, Any]:
    """Evaluate a class-aligned shared-space model with leave-one-subject-out CV.

    Parameters
    ----------
    subjects:
        Precomputed subject feature matrices.  No dataset-specific loading or
        window extraction is done here.
    config:
        Shared-space, target-calibration, classifier, and summary settings.
    outer_subjects:
        Optional subset of subject ids to hold out.  Defaults to every subject.
    fit_model:
        Optional classifier factory receiving ``(features, labels)``.  When
        omitted, the NeuRepTrace classifier registry is used.
    projection_transform:
        Optional adapter for projects whose decoding feature rows differ from
        the alignment feature rows, e.g. separate sensor/time windows.  The
        default assumes the fitted projection can be applied directly as
        ``(features - mean) @ projection``.
    label_shuffle_seed:
        Optional training-label shuffle control.  Labels are shuffled within
        each training subject and outer fold; target calibration labels are not
        shuffled.
    """

    config = _normalize_config(config or AlignmentBenchmarkConfig())
    prepared = tuple(_prepare_subject(subject) for subject in subjects)
    if len(prepared) < 3:
        raise ValueError("Leave-one-subject-out alignment benchmarks require at least three subjects.")
    subjects_by_id = {subject.subject_id: subject for subject in prepared}
    if len(subjects_by_id) != len(prepared):
        raise ValueError("Subject ids must be unique.")

    if outer_subjects is None:
        outer_ids = tuple(subject.subject_id for subject in prepared)
    else:
        outer_ids = tuple(outer_subjects)
        unknown = sorted((subject_id for subject_id in outer_ids if subject_id not in subjects_by_id), key=str)
        if unknown:
            raise ValueError(f"outer_subjects must be part of subjects: {unknown!r}.")

    transform = default_projection_transform if projection_transform is None else projection_transform
    fitted_model = _fit_model_factory(config, fit_model)
    outer_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    fold_results: list[AlignmentOuterFoldResult] = []

    for test_subject_id in outer_ids:
        if progress is not None:
            progress(f"START outer_test_subject={test_subject_id}")
        test_subject = subjects_by_id[test_subject_id]
        train_subjects = tuple(subject for subject in prepared if subject.subject_id != test_subject_id)
        fold_result = evaluate_alignment_outer_fold(
            train_subjects,
            test_subject,
            config=config,
            fit_model=fitted_model,
            projection_transform=transform,
            label_shuffle_seed=label_shuffle_seed,
        )
        outer_rows.append(fold_result.outer_row)
        prediction_rows.extend(fold_result.prediction_rows)
        fold_results.append(fold_result)
        if progress is not None:
            progress(
                "DONE outer_test_subject="
                f"{test_subject_id} balanced_accuracy={fold_result.outer_row['balanced_accuracy']:.4f}"
            )

    return {
        "outer": outer_rows,
        "predictions": prediction_rows,
        "group_summary": summarize_alignment_loso(outer_rows, config=config),
        "folds": fold_results,
    }


def evaluate_alignment_outer_fold(  # pylint: disable=too-many-locals
    train_subjects: Sequence[AlignmentSubjectData | _PreparedSubjectData],
    test_subject: AlignmentSubjectData | _PreparedSubjectData,
    *,
    config: AlignmentBenchmarkConfig | None = None,
    fit_model: Callable[[np.ndarray, np.ndarray], Any] | None = None,
    projection_transform: Callable[..., np.ndarray] | None = None,
    label_shuffle_seed: int | None = None,
) -> AlignmentOuterFoldResult:
    """Fit and score one held-out-subject alignment fold."""

    config = _normalize_config(config or AlignmentBenchmarkConfig())
    train = tuple(_prepare_subject(subject) for subject in train_subjects)
    test = _prepare_subject(test_subject)
    if len(train) < 2:
        raise ValueError("An outer fold requires at least two training subjects.")
    if any(subject.subject_id == test.subject_id for subject in train):
        raise ValueError("The test subject must not appear in train_subjects.")

    transform = default_projection_transform if projection_transform is None else projection_transform
    fitted_model = _fit_model_factory(config, fit_model)
    decode_labels_by_subject = {
        subject.subject_id: _training_labels(
            subject.labels,
            seed=label_shuffle_seed,
            test_subject=test.subject_id,
            train_subject=subject.subject_id,
        )
        for subject in train
    }
    alignment_features_by_subject = {subject.subject_id: subject.alignment_features for subject in train}
    alignment_labels_by_subject = {
        subject.subject_id: _source_alignment_labels(subject, decode_labels_by_subject[subject.subject_id])
        for subject in train
    }

    target_calibration_mode = config.target_calibration_mode
    if target_calibration_mode == "alignment_data":
        common_classes = _common_classes([*alignment_labels_by_subject.values(), test.alignment_labels])
        alignment_features_by_subject, alignment_labels_by_subject = _restrict_to_classes(
            alignment_features_by_subject,
            alignment_labels_by_subject,
            common_classes,
        )

    score_mask = np.ones(test.labels.shape[0], dtype=bool)
    calibration_mask = np.zeros(test.labels.shape[0], dtype=bool)
    if target_calibration_mode == "heldout_trials":
        calibration_mask = target_calibration_mask(test.labels, config.target_calibration_trials_per_class)
        score_mask = ~calibration_mask
        if test.alignment_features.shape[0] != test.labels.shape[0]:
            raise ValueError("heldout_trials target calibration requires alignment_features to match scored feature rows.")

    alignment_repetitions = config.n_repetitions_per_class
    if alignment_repetitions is None and target_calibration_mode == "heldout_trials" and config.sample_mode == "class_repetition":
        alignment_repetitions = config.target_calibration_trials_per_class

    alignment_model, class_alignment = _fit_alignment_model(
        alignment_features_by_subject,
        alignment_labels_by_subject,
        config=config,
        n_repetitions_per_class=alignment_repetitions,
    )

    train_matrix = np.vstack(
        [
            _transform_fitted_subject(alignment_model, subject, transform)
            for subject in train
        ]
    )
    train_labels = np.concatenate([decode_labels_by_subject[subject.subject_id] for subject in train])
    test_labels = test.labels[score_mask]

    test_matrix, target_transform, n_target_calibration_trials = _transform_target_subject(
        alignment_model,
        class_alignment,
        test,
        config,
        transform,
        score_mask=score_mask,
        calibration_mask=calibration_mask,
    )

    model_bundle = fit_window_model(
        train_matrix,
        train_labels,
        fit_model=fitted_model,
        components_pca=config.components_pca,
    )
    predictions, confidence_scores = predict_window_model(model_bundle, test_matrix)
    score_matrix, score_classes = _score_matrix(model_bundle, test_matrix)
    top_metrics, rank_rows = rank_metrics(test_labels, score_matrix, score_classes)

    accuracy = float(accuracy_score(test_labels, predictions)) if len(test_labels) else np.nan
    balanced = float(balanced_accuracy_score(test_labels, predictions)) if len(test_labels) else np.nan
    chance_classes = _chance_class_count(config, train_labels, test_labels)
    chance = 1.0 / chance_classes
    outer_row = {
        **_config_row(config),
        "test_subject": _label_value(test.subject_id),
        "train_subjects": ",".join(str(subject.subject_id) for subject in train),
        "n_train_subjects": len(train),
        "n_train_trials": int(train_matrix.shape[0]),
        "n_test_trials": int(test_labels.shape[0]),
        "n_target_calibration_trials": int(n_target_calibration_trials),
        "n_scored_trials": int(np.sum(score_mask)),
        "n_classes": int(np.unique(test_labels).size),
        "target_transform": target_transform,
        "alignment_actual_components": int(alignment_model.n_components),
        "alignment_rows": int(next(iter(class_alignment.aligned_by_subject.values())).shape[0]),
        "alignment_repetitions_per_class": class_alignment.n_repetitions_per_class,
        "alignment_classes": ",".join(str(_label_value(value)) for value in class_alignment.classes),
        "actual_components_pca": int(model_bundle.actual_components_pca),
        "pca_explained_variance_percent": model_bundle.explained_variance_percent,
        "chance_accuracy": chance,
        "chance_percent": 100.0 * chance,
        "accuracy": accuracy,
        "percent": 100.0 * accuracy if np.isfinite(accuracy) else np.nan,
        "balanced_accuracy": balanced,
        "balanced_percent": 100.0 * balanced if np.isfinite(balanced) else np.nan,
        "above_chance": bool(balanced > chance) if np.isfinite(balanced) else False,
        **top_metrics,
    }
    prediction_rows = _prediction_rows(
        test,
        config,
        test_labels,
        predictions,
        confidence_scores,
        rank_rows,
        score_mask=score_mask,
        target_transform=target_transform,
        actual_components=int(alignment_model.n_components),
        model_bundle=model_bundle,
    )
    return AlignmentOuterFoldResult(
        outer_row=outer_row,
        prediction_rows=prediction_rows,
        alignment_model=alignment_model,
        class_alignment=class_alignment,
        model_bundle=model_bundle,
    )


def default_projection_transform(
    features: Sequence[Sequence[float]] | np.ndarray,
    *,
    subject: AlignmentSubjectData | _PreparedSubjectData,
    projection: Sequence[Sequence[float]] | np.ndarray,
    projection_feature_mean: Sequence[float] | np.ndarray,
    projection_subject: AlignmentSubjectData | _PreparedSubjectData | None = None,
    feature_mean: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Apply a fitted subject or group projection directly to feature rows."""

    del subject, projection_subject
    matrix = _feature_matrix(features, name="features")
    projection_matrix = _feature_matrix(projection, name="projection")
    if matrix.shape[1] != projection_matrix.shape[0]:
        raise ValueError(f"features column count does not match projection rows: {matrix.shape[1]} != {projection_matrix.shape[0]}.")
    mean = np.asarray(projection_feature_mean if feature_mean is None else feature_mean, dtype=float).ravel()
    if mean.shape[0] != matrix.shape[1]:
        raise ValueError(f"feature mean length must match feature columns: {mean.shape[0]} != {matrix.shape[1]}.")
    return (matrix - mean) @ projection_matrix


def target_calibration_mask(labels: Sequence | np.ndarray, trials_per_class: int) -> np.ndarray:
    """Select the earliest target rows per class for labeled calibration."""

    labels = np.asarray(labels).ravel()
    trials_per_class = int(trials_per_class)
    if trials_per_class < 1:
        raise ValueError("trials_per_class must be positive for heldout_trials target calibration.")
    mask = np.zeros(labels.shape[0], dtype=bool)
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if indices.size <= trials_per_class:
            raise ValueError(
                f"Target class {label!r} has {indices.size} trials, which is not enough for "
                f"{trials_per_class} calibration trials plus at least one scored trial."
            )
        mask[indices[:trials_per_class]] = True
    return mask


def rank_metrics(
    true_labels: Sequence | np.ndarray,
    score_matrix: Sequence[Sequence[float]] | np.ndarray | None,
    classes: Sequence | np.ndarray | None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Compute top-k accuracy and per-row true-label ranks from class scores."""

    true_labels = np.asarray(true_labels).ravel()
    empty_rows: list[dict[str, Any]] = [{"true_label_rank": np.nan, "true_label_score": np.nan} for _ in true_labels]
    if score_matrix is None or classes is None:
        return _empty_rank_summary(), empty_rows
    scores = np.asarray(score_matrix, dtype=float)
    classes = np.asarray(classes)
    if scores.ndim != 2 or scores.shape[0] != true_labels.shape[0] or scores.shape[1] != classes.shape[0]:
        return _empty_rank_summary(), empty_rows

    order = np.argsort(scores, axis=1)[:, ::-1]
    rows: list[dict[str, Any]] = []
    ranks: list[float] = []
    for row_index, true_label in enumerate(true_labels):
        ranked_classes = classes[order[row_index]]
        matches = np.flatnonzero(ranked_classes == true_label)
        rank = float(matches[0] + 1) if matches.size else np.nan
        ranks.append(rank)
        true_column = np.flatnonzero(classes == true_label)
        row = {
            "true_label_rank": rank,
            "true_label_score": float(scores[row_index, true_column[0]]) if true_column.size else np.nan,
        }
        for rank_index, class_column in enumerate(order[row_index, :3], start=1):
            row[f"rank{rank_index}_label"] = _label_value(classes[class_column])
            row[f"rank{rank_index}_score"] = float(scores[row_index, class_column])
        rows.append(row)

    rank_array = np.asarray(ranks, dtype=float)
    finite = rank_array[np.isfinite(rank_array)]
    summary = {
        "top2_accuracy": float(np.mean(rank_array <= 2)) if rank_array.size else np.nan,
        "top2_percent": float(100.0 * np.mean(rank_array <= 2)) if rank_array.size else np.nan,
        "top3_accuracy": float(np.mean(rank_array <= 3)) if rank_array.size else np.nan,
        "top3_percent": float(100.0 * np.mean(rank_array <= 3)) if rank_array.size else np.nan,
        "mean_true_label_rank": float(np.mean(finite)) if finite.size else np.nan,
    }
    return summary, rows


def summarize_alignment_loso(outer_rows: Sequence[Mapping[str, Any]], *, config: AlignmentBenchmarkConfig | None = None) -> list[dict[str, Any]]:
    """Summarize held-out-subject alignment scores."""

    if not outer_rows:
        return []
    config = _normalize_config(config or AlignmentBenchmarkConfig())
    balanced = _finite_values(outer_rows, "balanced_accuracy")
    raw = _finite_values(outer_rows, "accuracy")
    top2 = _finite_values(outer_rows, "top2_accuracy")
    top3 = _finite_values(outer_rows, "top3_accuracy")
    ranks = _finite_values(outer_rows, "mean_true_label_rank")
    chance = float(outer_rows[0]["chance_accuracy"])
    differences = balanced - chance
    return [
        {
            **_config_row(config),
            "n_outer_folds": len(outer_rows),
            "n_test_subjects": len(outer_rows),
            "chance_accuracy": chance,
            "chance_percent": 100.0 * chance,
            "accuracy_mean": _nanmean(raw),
            "accuracy_median": _nanmedian(raw),
            "accuracy_sem": _sem(raw),
            "percent_mean": _percent(_nanmean(raw)),
            "balanced_accuracy_mean": _nanmean(balanced),
            "balanced_accuracy_median": _nanmedian(balanced),
            "balanced_accuracy_sem": _sem(balanced),
            "balanced_percent_mean": _percent(_nanmean(balanced)),
            "balanced_percent_median": _percent(_nanmedian(balanced)),
            "balanced_percent_sem": _percent(_sem(balanced)),
            "top2_accuracy_mean": _nanmean(top2),
            "top2_percent_mean": _percent(_nanmean(top2)),
            "top3_accuracy_mean": _nanmean(top3),
            "top3_percent_mean": _percent(_nanmean(top3)),
            "mean_true_label_rank_mean": _nanmean(ranks),
            "mean_true_label_rank_sem": _sem(ranks),
            "chance_mean_rank": 0.5 * ((1.0 / chance) + 1.0),
            "mean_above_chance": _nanmean(differences),
            "percent_above_chance": _percent(_nanmean(differences)),
            "subjects_above_chance": int(np.sum(differences > 0)) if differences.size else 0,
            "subjects_total": int(differences.size),
            "subjects_at_or_below_chance": int(np.sum(differences <= 0)) if differences.size else 0,
            "one_sided_exact_sign_p_value": _one_sided_exact_sign_p_value(differences),
            "one_sided_signflip_p_value": _one_sided_signflip_p_value(
                differences,
                n_permutations=config.signflip_permutations,
                seed=config.signflip_seed,
            ),
        }
    ]


def _fit_alignment_model(features_by_subject, labels_by_subject, *, config, n_repetitions_per_class):
    if config.method == "hyperalignment":
        return fit_class_hyperalignment(
            features_by_subject,
            labels_by_subject,
            sample_mode=config.sample_mode,
            n_repetitions_per_class=n_repetitions_per_class,
            n_components=config.n_components,
            n_iterations=config.hyperalignment_iterations,
            initialization=config.hyperalignment_initialization,
        )
    if config.method == "mcca":
        return fit_class_mcca(
            features_by_subject,
            labels_by_subject,
            sample_mode=config.sample_mode,
            n_repetitions_per_class=n_repetitions_per_class,
            n_components=config.n_components,
            regularization=config.mcca_regularization,
            subject_pca_components=config.mcca_subject_pca_components,
        )
    raise ValueError(f"Unsupported alignment method: {config.method}.")


def _transform_fitted_subject(model, subject: _PreparedSubjectData, transform: Callable[..., np.ndarray]) -> np.ndarray:
    projection = model.projections[subject.subject_id]
    return transform(
        subject.features,
        subject=subject,
        projection=projection.projection,
        projection_feature_mean=projection.feature_mean,
        projection_subject=subject,
    )


def _transform_target_subject(
    model,
    class_alignment,
    test: _PreparedSubjectData,
    config: AlignmentBenchmarkConfig,
    transform: Callable[..., np.ndarray],
    *,
    score_mask: np.ndarray,
    calibration_mask: np.ndarray,
) -> tuple[np.ndarray, str, int]:
    if config.target_calibration_mode == "alignment_data":
        if not (test.alignment_features_explicit and test.alignment_labels_explicit):
            raise ValueError("alignment_data target calibration requires explicit target alignment_features and alignment_labels.")
        target_aligned = class_alignment_matrix(
            test.alignment_features,
            test.alignment_labels,
            classes=class_alignment.classes,
            sample_mode=class_alignment.sample_mode,
            n_repetitions_per_class=class_alignment.n_repetitions_per_class,
        )
        projection = _fit_target_projection(target_aligned, model, config)
        transformed = transform(
            test.features[score_mask],
            subject=test,
            projection=projection.projection,
            projection_feature_mean=projection.feature_mean,
            projection_subject=test,
        )
        transformed = _add_mcca_template_mean_if_needed(transformed, projection)
        return transformed, "alignment_data_calibrated", _count_labels_in_classes(test.alignment_labels, class_alignment.classes)

    if config.target_calibration_mode == "heldout_trials":
        target_aligned = class_alignment_matrix(
            test.alignment_features[calibration_mask],
            test.labels[calibration_mask],
            classes=class_alignment.classes,
            sample_mode=class_alignment.sample_mode,
            n_repetitions_per_class=class_alignment.n_repetitions_per_class,
        )
        projection = _fit_target_projection(target_aligned, model, config)
        transformed = transform(
            test.features[score_mask],
            subject=test,
            projection=projection.projection,
            projection_feature_mean=projection.feature_mean,
            projection_subject=test,
        )
        transformed = _add_mcca_template_mean_if_needed(transformed, projection)
        return transformed, "target_calibrated", int(np.sum(calibration_mask))

    if model.group_projection is None or model.group_feature_mean is None:
        raise ValueError("A group alignment projection is unavailable for the held-out subject.")
    target_mean = np.mean(test.features, axis=0) if config.target_centering == "target_unsupervised" else None
    transformed = transform(
        test.features[score_mask],
        subject=test,
        projection=model.group_projection,
        projection_feature_mean=model.group_feature_mean,
        projection_subject=None,
        feature_mean=target_mean,
    )
    target_transform = "group_average" if config.method == "hyperalignment" else "group_projection"
    return transformed, target_transform, 0


def _fit_target_projection(target_aligned: np.ndarray, model, config: AlignmentBenchmarkConfig):
    if config.method == "hyperalignment":
        return fit_projection_to_hyperalignment(target_aligned, template=model.template)
    regularization = config.target_projection_regularization
    if regularization is None:
        regularization = config.mcca_regularization
    return fit_target_mcca_projection(target_aligned, model, regularization=regularization)


def _add_mcca_template_mean_if_needed(transformed: np.ndarray, projection) -> np.ndarray:
    if isinstance(projection, TargetMCCAProjection):
        return projection.add_template_mean(transformed)
    return transformed


def _score_matrix(model_bundle, features: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    transformed = transform_window_features(model_bundle, features)
    model = model_bundle.model
    classes = _model_classes(model, model_bundle.train_labels)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(transformed), dtype=float)
    elif hasattr(model, "predict_proba"):
        scores = np.asarray(model.predict_proba(transformed), dtype=float)
    else:
        predictions = np.asarray(model.predict(transformed))
        scores = np.zeros((predictions.shape[0], classes.shape[0]), dtype=float)
        for row_index, prediction in enumerate(predictions):
            matches = np.flatnonzero(classes == prediction)
            if matches.size:
                scores[row_index, matches[0]] = 1.0
    if scores.ndim == 1 and classes.shape[0] == 2:
        scores = np.column_stack((-scores, scores))
    if scores.ndim != 2 or scores.shape[1] != classes.shape[0]:
        if hasattr(model, "predict_proba"):
            scores = np.asarray(model.predict_proba(transformed), dtype=float)
        else:
            return None, None
    return scores, classes


def _model_classes(model, fallback_labels: np.ndarray) -> np.ndarray:
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        try:
            classes = getattr(list(model.named_steps.values())[-1], "classes_", None)
        except IndexError:
            classes = None
    if classes is None:
        return np.unique(fallback_labels)
    return np.asarray(classes)


def _prediction_rows(
    test: _PreparedSubjectData,
    config: AlignmentBenchmarkConfig,
    test_labels: np.ndarray,
    predictions: np.ndarray,
    confidence_scores: np.ndarray,
    rank_rows: Sequence[Mapping[str, Any]],
    *,
    score_mask: np.ndarray,
    target_transform: str,
    actual_components: int,
    model_bundle,
) -> list[dict[str, Any]]:
    trial_indices = np.flatnonzero(score_mask)
    rows: list[dict[str, Any]] = []
    for output_index, (trial_index, truth, prediction, confidence) in enumerate(
        zip(trial_indices, test_labels, predictions, confidence_scores, strict=True)
    ):
        rows.append(
            {
                **_config_row(config),
                "test_subject": _label_value(test.subject_id),
                "trial_index": int(trial_index),
                "true_label": _label_value(truth),
                "predicted_label": _label_value(prediction),
                "correct": bool(prediction == truth),
                "score": float(confidence),
                "target_transform": target_transform,
                "alignment_actual_components": int(actual_components),
                "actual_components_pca": int(model_bundle.actual_components_pca),
                **dict(rank_rows[output_index]),
            }
        )
    return rows


def _prepare_subject(subject: AlignmentSubjectData | _PreparedSubjectData) -> _PreparedSubjectData:
    if isinstance(subject, _PreparedSubjectData):
        return subject
    features = _feature_matrix(subject.features, name=f"features[{subject.subject_id!r}]")
    labels = _label_vector(subject.labels, expected_length=features.shape[0], name=f"labels[{subject.subject_id!r}]")
    alignment_features_explicit = subject.alignment_features is not None
    alignment_labels_explicit = subject.alignment_labels is not None
    alignment_features = features if subject.alignment_features is None else _feature_matrix(
        subject.alignment_features,
        name=f"alignment_features[{subject.subject_id!r}]",
    )
    alignment_uses_decode_labels = not alignment_labels_explicit
    if alignment_uses_decode_labels:
        if alignment_features.shape[0] != labels.shape[0]:
            raise ValueError("alignment_labels are required when alignment_features have a different row count from features.")
        alignment_labels = labels
    else:
        alignment_labels = _label_vector(
            subject.alignment_labels,
            expected_length=alignment_features.shape[0],
            name=f"alignment_labels[{subject.subject_id!r}]",
        )
    return _PreparedSubjectData(
        subject_id=subject.subject_id,
        features=features,
        labels=labels,
        alignment_features=alignment_features,
        alignment_labels=alignment_labels,
        alignment_features_explicit=alignment_features_explicit,
        alignment_labels_explicit=alignment_labels_explicit,
        alignment_uses_decode_labels=alignment_uses_decode_labels,
    )


def _normalize_config(config: AlignmentBenchmarkConfig) -> AlignmentBenchmarkConfig:
    method = _normalize_choice(config.method, ALIGNMENT_METHODS, "alignment method")
    sample_mode = str(config.sample_mode).strip().lower().replace("-", "_")
    target_mode = _normalize_choice(config.target_calibration_mode, TARGET_CALIBRATION_MODES, "target calibration mode")
    target_centering = _normalize_choice(config.target_centering, TARGET_CENTERING_MODES, "target centering")
    target_trials = int(config.target_calibration_trials_per_class)
    if target_trials < 0:
        raise ValueError("target_calibration_trials_per_class must be non-negative.")
    if target_trials > 0 and target_mode == "none":
        target_mode = "heldout_trials"
    if target_mode == "heldout_trials" and target_trials < 1:
        raise ValueError("heldout_trials target calibration requires target_calibration_trials_per_class >= 1.")
    if config.mcca_regularization < 0:
        raise ValueError("mcca_regularization must be non-negative.")
    if config.target_projection_regularization is not None and config.target_projection_regularization < 0:
        raise ValueError("target_projection_regularization must be non-negative.")
    if config.hyperalignment_iterations < 1:
        raise ValueError("hyperalignment_iterations must be positive.")
    if config.chance_classes is not None and int(config.chance_classes) < 1:
        raise ValueError("chance_classes must be positive or None.")
    if config.signflip_permutations < 0:
        raise ValueError("signflip_permutations must be non-negative.")
    return replace(
        config,
        method=method,
        sample_mode=sample_mode,
        target_calibration_mode=target_mode,
        target_calibration_trials_per_class=target_trials,
        target_centering=target_centering,
        signflip_permutations=int(config.signflip_permutations),
    )


def _normalize_choice(value: str, choices: Sequence[str], name: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in choices:
        raise ValueError(f"Unsupported {name}: {value}. Supported values: {', '.join(choices)}.")
    return normalized


def _fit_model_factory(config: AlignmentBenchmarkConfig, fit_model: Callable[[np.ndarray, np.ndarray], Any] | None):
    if fit_model is not None:
        return fit_model
    classifier_param = get_default_classifier_param(config.classifier) if config.classifier_param is None else config.classifier_param

    def factory(features: np.ndarray, labels: np.ndarray):
        return train_multiclass_classifier(
            features,
            labels,
            config.classifier,
            classifier_param,
            random_state=config.random_state,
        )

    return factory


def _source_alignment_labels(subject: _PreparedSubjectData, shuffled_decode_labels: np.ndarray) -> np.ndarray:
    if subject.alignment_uses_decode_labels:
        return shuffled_decode_labels
    return subject.alignment_labels


def _training_labels(labels: np.ndarray, *, seed: int | None, test_subject: Hashable, train_subject: Hashable) -> np.ndarray:
    labels = np.asarray(labels).copy()
    if seed is None:
        return labels
    rng = np.random.default_rng(np.random.SeedSequence([int(seed) & 0xFFFFFFFF, _seed_value(test_subject), _seed_value(train_subject)]))
    rng.shuffle(labels)
    return labels


def _seed_value(value: Hashable) -> int:
    try:
        return int(value) & 0xFFFFFFFF
    except (TypeError, ValueError):
        return crc32(repr(value).encode("utf-8")) & 0xFFFFFFFF


def _common_classes(label_arrays: Sequence[Sequence | np.ndarray]) -> np.ndarray:
    label_sets = [set(np.asarray(labels).ravel().tolist()) for labels in label_arrays]
    common = sorted(set.intersection(*label_sets), key=str) if label_sets else []
    if len(common) < 2:
        raise ValueError("Alignment-data target calibration requires at least two classes shared by source and target alignment rows.")
    return np.asarray(common)


def _restrict_to_classes(features_by_subject, labels_by_subject, classes):
    classes = np.asarray(classes).ravel()
    filtered_features = {}
    filtered_labels = {}
    for subject_id, labels in labels_by_subject.items():
        label_array = np.asarray(labels).ravel()
        mask = np.isin(label_array, classes)
        filtered_features[subject_id] = np.asarray(features_by_subject[subject_id], dtype=float)[mask]
        filtered_labels[subject_id] = label_array[mask]
    return filtered_features, filtered_labels


def _count_labels_in_classes(labels, classes) -> int:
    return int(np.sum(np.isin(np.asarray(labels).ravel(), np.asarray(classes).ravel())))


def _chance_class_count(config: AlignmentBenchmarkConfig, train_labels: np.ndarray, test_labels: np.ndarray) -> int:
    if config.chance_classes is not None:
        return int(config.chance_classes)
    return int(np.unique(np.concatenate([train_labels, test_labels])).size)


def _config_row(config: AlignmentBenchmarkConfig) -> dict[str, Any]:
    return {
        "alignment_method": config.method,
        "alignment_sample_mode": config.sample_mode,
        "alignment_requested_components": config.n_components,
        "n_repetitions_per_class": config.n_repetitions_per_class,
        "hyperalignment_iterations": config.hyperalignment_iterations if config.method == "hyperalignment" else "",
        "hyperalignment_initialization": config.hyperalignment_initialization if config.method == "hyperalignment" else "",
        "mcca_regularization": config.mcca_regularization if config.method == "mcca" else "",
        "mcca_subject_pca_components": config.mcca_subject_pca_components if config.method == "mcca" else "",
        "target_calibration_mode": config.target_calibration_mode,
        "target_calibration_trials_per_class": config.target_calibration_trials_per_class,
        "target_centering": config.target_centering,
        "target_projection_regularization": config.target_projection_regularization,
        "classifier": config.classifier,
        "classifier_param": config.classifier_param,
        "components_pca": config.components_pca,
    }


def _empty_rank_summary() -> dict[str, float]:
    return {
        "top2_accuracy": np.nan,
        "top2_percent": np.nan,
        "top3_accuracy": np.nan,
        "top3_percent": np.nan,
        "mean_true_label_rank": np.nan,
    }


def _finite_values(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    values = np.asarray([float(row.get(key, np.nan)) for row in rows], dtype=float)
    return values[np.isfinite(values)]


def _nanmean(values: Sequence[float] | np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.mean(values)) if values.size else np.nan


def _nanmedian(values: Sequence[float] | np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.median(values)) if values.size else np.nan


def _sem(values: Sequence[float] | np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan
    if values.size == 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def _percent(value: float) -> float:
    return float(100.0 * value) if np.isfinite(value) else np.nan


def _one_sided_exact_sign_p_value(differences: Sequence[float] | np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    nonzero = differences[np.isfinite(differences) & (differences != 0)]
    if nonzero.size == 0:
        return np.nan
    positives = int(np.sum(nonzero > 0))
    tail = sum(comb(nonzero.size, k) for k in range(positives, nonzero.size + 1))
    return float(tail / (2**nonzero.size))


def _one_sided_signflip_p_value(differences: Sequence[float] | np.ndarray, *, n_permutations: int, seed: int | None) -> float:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if differences.size == 0 or n_permutations <= 0:
        return np.nan
    observed = float(np.mean(differences))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(int(n_permutations), differences.size))
    null_values = np.mean(signs * differences[None, :], axis=1)
    return float((np.sum(null_values >= observed) + 1.0) / (int(n_permutations) + 1.0))


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


def _label_value(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "ALIGNMENT_METHODS",
    "TARGET_CALIBRATION_MODES",
    "TARGET_CENTERING_MODES",
    "AlignmentBenchmarkConfig",
    "AlignmentOuterFoldResult",
    "AlignmentSubjectData",
    "default_projection_transform",
    "evaluate_alignment_loso",
    "evaluate_alignment_outer_fold",
    "rank_metrics",
    "summarize_alignment_loso",
    "target_calibration_mask",
]
