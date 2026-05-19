"""Generic leave-one-subject-out cross-subject decoding utilities."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from math import comb
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from neureptrace.decoding.windowed import (
    FitModel,
    WindowedModelBundle,
    fit_window_model,
    transform_window_features,
)

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class SubjectFeatureSet:
    """Precomputed trial/window features for one subject.

    Dataset-specific packages own loading, event parsing, and window extraction.
    NeuRepTrace only requires rows of features, matching labels, and a stable
    subject identifier.
    """

    subject: Hashable
    features: Sequence[Sequence[float]] | np.ndarray
    labels: Sequence | np.ndarray
    trial_ids: Sequence | np.ndarray | None = None
    metadata: Any | None = None


@dataclass(frozen=True)
class CrossSubjectCandidate:
    """One model/preprocessing candidate for cross-subject evaluation."""

    name: str
    fit_model: FitModel
    components_pca: int | float = float("inf")
    train_window: tuple[float, float] | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class CrossSubjectFittedModel:
    """A fitted candidate plus fold provenance."""

    candidate: CrossSubjectCandidate
    candidate_index: int
    model_bundle: WindowedModelBundle
    train_subjects: tuple[Hashable, ...]
    train_class_counts: Mapping[Any, int]
    chance_classes: int
    label_shuffle_control: bool = False
    label_shuffle_seed: int | str = ""


@dataclass(frozen=True)
class CrossSubjectEvaluationResult:
    """Artifacts from a fixed-candidate LOSO evaluation."""

    outer: list[dict[str, Any]]
    predictions: list[dict[str, Any]]
    group_summary: list[dict[str, Any]]


@dataclass(frozen=True)
class NestedCrossSubjectEvaluationResult:
    """Artifacts from nested LOSO model selection and held-out scoring."""

    outer: list[dict[str, Any]]
    inner_validation: list[dict[str, Any]]
    selected: list[dict[str, Any]]
    predictions: list[dict[str, Any]]
    group_summary: list[dict[str, Any]]


def leave_one_subject_out_decoding(
    feature_sets: Sequence[SubjectFeatureSet],
    *,
    fit_model: FitModel | None = None,
    candidate: CrossSubjectCandidate | None = None,
    components_pca: int | float = float("inf"),
    train_window: tuple[float, float] | None = None,
    chance_classes: int | None = None,
    include_predictions: bool = True,
    progress: ProgressCallback | None = None,
) -> CrossSubjectEvaluationResult:
    """Evaluate one candidate with leave-one-subject-out cross-validation.

    Callers can either pass a full :class:`CrossSubjectCandidate` or pass
    ``fit_model`` plus optional PCA/window metadata. Dataset-specific wrappers
    should prepare :class:`SubjectFeatureSet` objects and delegate the generic
    fold bookkeeping here.
    """

    feature_sets = _validate_feature_sets(feature_sets)
    if len(feature_sets) < 2:
        raise ValueError("At least two subjects are required for leave-one-subject-out decoding.")
    candidate = _coerce_candidate(
        candidate,
        fit_model=fit_model,
        components_pca=components_pca,
        train_window=train_window,
    )
    chance_classes = _resolve_chance_classes(feature_sets, chance_classes)

    outer_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for test_set in feature_sets:
        if progress is not None:
            progress(f"START outer_test_subject={test_set.subject}")
        train_sets = tuple(feature_set for feature_set in feature_sets if feature_set.subject != test_set.subject)
        fitted_model = fit_cross_subject_model(
            train_sets,
            candidate,
            candidate_index=1,
            chance_classes=chance_classes,
        )
        outer_row, fold_predictions = score_cross_subject_model(
            fitted_model,
            test_set,
            include_predictions=include_predictions,
        )
        outer_rows.append(outer_row)
        prediction_rows.extend(fold_predictions)
        if progress is not None:
            progress(f"DONE outer_test_subject={test_set.subject} balanced_accuracy={outer_row['balanced_accuracy']:.4f}")

    return CrossSubjectEvaluationResult(
        outer=outer_rows,
        predictions=prediction_rows,
        group_summary=summarize_cross_subject_folds(outer_rows),
    )


def nested_leave_one_subject_out_decoding(
    feature_sets: Sequence[SubjectFeatureSet],
    *,
    candidates: Sequence[CrossSubjectCandidate],
    outer_subjects: Sequence[Hashable] | None = None,
    selection_metric: str = "balanced_accuracy",
    chance_classes: int | None = None,
    include_predictions: bool = True,
    progress: ProgressCallback | None = None,
) -> NestedCrossSubjectEvaluationResult:
    """Run nested LOSO candidate selection and untouched outer-fold scoring.

    For each outer subject, candidates are scored on inner LOSO folds drawn only
    from the outer-training subjects. The best mean inner score is then refit on
    all outer-training subjects and evaluated once on the untouched outer subject.
    """

    feature_sets = _validate_feature_sets(feature_sets)
    if len(feature_sets) < 3:
        raise ValueError("At least three subjects are required for nested leave-one-subject-out decoding.")
    candidates = _validate_candidates(candidates)
    selection_metric = _normalize_selection_metric(selection_metric)
    chance_classes = _resolve_chance_classes(feature_sets, chance_classes)
    outer_subjects = _normalize_outer_subjects(feature_sets, outer_subjects)
    subject_to_set = {feature_set.subject: feature_set for feature_set in feature_sets}

    outer_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for outer_subject in outer_subjects:
        if progress is not None:
            progress(f"START outer_test_subject={outer_subject}")
        outer_train_sets = tuple(feature_set for feature_set in feature_sets if feature_set.subject != outer_subject)
        outer_inner_rows: list[dict[str, Any]] = []

        for candidate_index, candidate in enumerate(candidates, start=1):
            for validation_set in outer_train_sets:
                inner_train_sets = tuple(feature_set for feature_set in outer_train_sets if feature_set.subject != validation_set.subject)
                fitted_model = fit_cross_subject_model(
                    inner_train_sets,
                    candidate,
                    candidate_index=candidate_index,
                    chance_classes=chance_classes,
                )
                inner_row, _fold_predictions = score_cross_subject_model(
                    fitted_model,
                    validation_set,
                    include_predictions=False,
                )
                inner_row.update(
                    {
                        "selection_mode": "nested_loso",
                        "selection_metric": selection_metric,
                        "outer_fold": _row_value(outer_subject),
                        "outer_test_subject": _row_value(outer_subject),
                        "inner_fold": _row_value(validation_set.subject),
                        "inner_validation_subject": _row_value(validation_set.subject),
                    }
                )
                outer_inner_rows.append(inner_row)
                if progress is not None:
                    progress(
                        "DONE inner_validation "
                        f"outer_test_subject={outer_subject} "
                        f"candidate={candidate_index}/{len(candidates)} "
                        f"validation_subject={validation_set.subject}"
                    )

        selected_row = _select_nested_candidate(outer_inner_rows, candidates, selection_metric=selection_metric)
        selected_rows.append(selected_row)
        selected_candidate_index = int(selected_row["selected_candidate_index"])
        selected_candidate = candidates[selected_candidate_index - 1]
        fitted_model = fit_cross_subject_model(
            outer_train_sets,
            selected_candidate,
            candidate_index=selected_candidate_index,
            chance_classes=chance_classes,
        )
        outer_row, fold_predictions = score_cross_subject_model(
            fitted_model,
            subject_to_set[outer_subject],
            include_predictions=include_predictions,
        )
        _add_selected_fields(outer_row, selected_row)
        for prediction_row in fold_predictions:
            _add_selected_fields(prediction_row, selected_row)
        inner_rows.extend(outer_inner_rows)
        outer_rows.append(outer_row)
        prediction_rows.extend(fold_predictions)
        if progress is not None:
            progress(
                f"DONE outer_test_subject={outer_subject} "
                f"selected_candidate={selected_candidate_index} "
                f"balanced_accuracy={outer_row['balanced_accuracy']:.4f}"
            )

    return NestedCrossSubjectEvaluationResult(
        outer=outer_rows,
        inner_validation=inner_rows,
        selected=selected_rows,
        predictions=prediction_rows,
        group_summary=summarize_cross_subject_folds(outer_rows),
    )


def fit_cross_subject_model(
    train_sets: Sequence[SubjectFeatureSet],
    candidate: CrossSubjectCandidate,
    *,
    candidate_index: int = 1,
    chance_classes: int | None = None,
    label_shuffle_seed: int | None = None,
    label_shuffle_context: Sequence[Hashable] = (),
) -> CrossSubjectFittedModel:
    """Fit a candidate on a collection of training subjects."""

    train_sets = _validate_feature_sets(train_sets)
    if not train_sets:
        raise ValueError("At least one training subject is required.")
    candidate = _validate_candidate(candidate)
    chance_classes = _resolve_chance_classes(train_sets, chance_classes)

    train_features = np.vstack([feature_set.features for feature_set in train_sets])
    label_arrays = [
        _training_labels(
            feature_set,
            label_shuffle_seed=label_shuffle_seed,
            label_shuffle_context=(candidate_index, *tuple(label_shuffle_context)),
        )
        for feature_set in train_sets
    ]
    train_labels = np.concatenate(label_arrays)
    model_bundle = fit_window_model(
        train_features,
        train_labels,
        fit_model=candidate.fit_model,
        components_pca=candidate.components_pca,
        train_window=candidate.train_window,
    )
    return CrossSubjectFittedModel(
        candidate=candidate,
        candidate_index=int(candidate_index),
        model_bundle=model_bundle,
        train_subjects=tuple(feature_set.subject for feature_set in train_sets),
        train_class_counts=dict(Counter(_row_value(label) for label in train_labels.tolist())),
        chance_classes=int(chance_classes),
        label_shuffle_control=label_shuffle_seed is not None,
        label_shuffle_seed="" if label_shuffle_seed is None else int(label_shuffle_seed),
    )


def score_cross_subject_model(
    fitted_model: CrossSubjectFittedModel,
    test_set: SubjectFeatureSet,
    *,
    include_predictions: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score a fitted cross-subject model on one held-out subject."""

    test_set = _validate_subject_feature_set(test_set)
    test_features = np.asarray(test_set.features, dtype=float)
    test_labels = np.asarray(test_set.labels).ravel()
    predictions, class_scores, score_classes = _predict_with_class_scores(fitted_model.model_bundle, test_features)
    rank_metrics = _ranked_label_metrics(test_labels, class_scores, score_classes)
    accuracy = float(accuracy_score(test_labels, predictions))
    balanced_accuracy = float(balanced_accuracy_score(test_labels, predictions))
    chance_accuracy = 1.0 / float(fitted_model.chance_classes)
    test_class_counts = Counter(_row_value(label) for label in test_labels.tolist())

    outer_row: dict[str, Any] = {
        "outer_fold": _row_value(test_set.subject),
        "test_subject": _row_value(test_set.subject),
        "train_subjects": _format_subjects(fitted_model.train_subjects),
        "n_train_subjects": len(fitted_model.train_subjects),
        "n_test_subjects": 1,
        "accuracy": accuracy,
        "percent": 100.0 * accuracy,
        "balanced_accuracy": balanced_accuracy,
        "balanced_percent": 100.0 * balanced_accuracy,
        "top2_accuracy": rank_metrics["top2_accuracy"],
        "top2_percent": 100.0 * rank_metrics["top2_accuracy"],
        "top3_accuracy": rank_metrics["top3_accuracy"],
        "top3_percent": 100.0 * rank_metrics["top3_accuracy"],
        "mean_true_label_rank": rank_metrics["mean_true_label_rank"],
        "median_true_label_rank": rank_metrics["median_true_label_rank"],
        "chance_accuracy": chance_accuracy,
        "chance_percent": 100.0 * chance_accuracy,
        "top2_chance_accuracy": min(2.0 * chance_accuracy, 1.0),
        "top2_chance_percent": min(200.0 * chance_accuracy, 100.0),
        "top3_chance_accuracy": min(3.0 * chance_accuracy, 1.0),
        "top3_chance_percent": min(300.0 * chance_accuracy, 100.0),
        "chance_mean_rank": 0.5 * (fitted_model.chance_classes + 1),
        "above_chance": bool(balanced_accuracy > chance_accuracy),
        "n_train_trials": int(fitted_model.model_bundle.train_labels.shape[0]),
        "n_test_trials": int(test_labels.shape[0]),
        "n_train_classes": int(len(fitted_model.train_class_counts)),
        "n_test_classes": int(len(test_class_counts)),
        "min_train_trials_per_class": int(min(fitted_model.train_class_counts.values())),
        "min_test_trials_per_class": int(min(test_class_counts.values())),
        "actual_components_pca": fitted_model.model_bundle.actual_components_pca,
        "pca_explained_variance_percent": fitted_model.model_bundle.explained_variance_percent,
        "label_shuffle_control": bool(fitted_model.label_shuffle_control),
        "label_shuffle_seed": fitted_model.label_shuffle_seed,
    }
    _add_candidate_fields(outer_row, fitted_model.candidate, fitted_model.candidate_index)

    prediction_rows: list[dict[str, Any]] = []
    if include_predictions:
        trial_ids = _trial_ids(test_set)
        for trial_number, trial_id, true_label, predicted_label, true_label_rank in zip(
            range(1, test_labels.shape[0] + 1),
            trial_ids,
            test_labels,
            predictions,
            rank_metrics["true_label_ranks"],
            strict=True,
        ):
            prediction_row: dict[str, Any] = {
                "outer_fold": _row_value(test_set.subject),
                "test_subject": _row_value(test_set.subject),
                "trial": _row_value(trial_id),
                "trial_index": _row_value(trial_id),
                "trial_number": int(trial_number),
                "true_label": _row_value(true_label),
                "predicted_label": _row_value(predicted_label),
                "correct": bool(predicted_label == true_label),
                "true_label_rank": float(true_label_rank) if np.isfinite(true_label_rank) else np.nan,
                "top2_correct": bool(np.isfinite(true_label_rank) and true_label_rank <= 2),
                "top3_correct": bool(np.isfinite(true_label_rank) and true_label_rank <= 3),
            }
            _add_candidate_fields(prediction_row, fitted_model.candidate, fitted_model.candidate_index)
            prediction_rows.append(prediction_row)
    return outer_row, prediction_rows


def summarize_cross_subject_folds(outer_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize LOSO fold scores across held-out subjects."""

    outer_rows = tuple(outer_rows)
    if not outer_rows:
        return []

    accuracy = _row_float_values(outer_rows, "accuracy")
    balanced = _row_float_values(outer_rows, "balanced_accuracy")
    chance = _row_float_values(outer_rows, "chance_accuracy")
    top2 = _row_float_values(outer_rows, "top2_accuracy")
    top3 = _row_float_values(outer_rows, "top3_accuracy")
    mean_ranks = _row_float_values(outer_rows, "mean_true_label_rank")
    differences = balanced - chance
    return [
        {
            "n_outer_folds": len(outer_rows),
            "n_test_subjects": len({_row_value(row["test_subject"]) for row in outer_rows}),
            "accuracy_mean": float(np.mean(accuracy)),
            "accuracy_median": float(np.median(accuracy)),
            "accuracy_sem": _sem(accuracy),
            "percent_mean": float(100.0 * np.mean(accuracy)),
            "balanced_accuracy_mean": float(np.mean(balanced)),
            "balanced_accuracy_median": float(np.median(balanced)),
            "balanced_accuracy_sem": _sem(balanced),
            "balanced_percent_mean": float(100.0 * np.mean(balanced)),
            "balanced_percent_median": float(100.0 * np.median(balanced)),
            "balanced_percent_sem": float(100.0 * _sem(balanced)),
            "top2_accuracy_mean": _nanmean_or_nan(top2),
            "top2_percent_mean": _percent_nanmean_or_nan(top2),
            "top3_accuracy_mean": _nanmean_or_nan(top3),
            "top3_percent_mean": _percent_nanmean_or_nan(top3),
            "mean_true_label_rank_mean": _nanmean_or_nan(mean_ranks),
            "mean_true_label_rank_sem": _sem_or_nan(mean_ranks),
            "chance_accuracy_mean": float(np.mean(chance)),
            "chance_percent_mean": float(100.0 * np.mean(chance)),
            "mean_above_chance": float(np.mean(differences)),
            "percent_above_chance": float(100.0 * np.mean(differences)),
            "subjects_above_chance": int(np.sum(differences > 0.0)),
            "subjects_total": int(np.sum(np.isfinite(differences))),
            "subjects_at_or_below_chance": int(np.sum(differences <= 0.0)),
            "one_sided_exact_sign_p_value": _one_sided_exact_sign_p_value(differences),
        }
    ]


def _coerce_candidate(
    candidate: CrossSubjectCandidate | None,
    *,
    fit_model: FitModel | None,
    components_pca: int | float,
    train_window: tuple[float, float] | None,
) -> CrossSubjectCandidate:
    if candidate is not None:
        if fit_model is not None:
            raise ValueError("Pass either candidate or fit_model, not both.")
        return _validate_candidate(candidate)
    if fit_model is None:
        raise ValueError("Either candidate or fit_model is required.")
    return _validate_candidate(
        CrossSubjectCandidate(
            name="candidate",
            fit_model=fit_model,
            components_pca=components_pca,
            train_window=train_window,
        )
    )


def _validate_feature_sets(feature_sets: Sequence[SubjectFeatureSet]) -> tuple[SubjectFeatureSet, ...]:
    validated = tuple(_validate_subject_feature_set(feature_set) for feature_set in feature_sets)
    subjects = [feature_set.subject for feature_set in validated]
    if len(set(subjects)) != len(subjects):
        raise ValueError("Subject identifiers must be unique.")
    return validated


def _validate_subject_feature_set(feature_set: SubjectFeatureSet) -> SubjectFeatureSet:
    if not isinstance(feature_set, SubjectFeatureSet):
        raise TypeError("feature_sets must contain SubjectFeatureSet instances.")
    try:
        hash(feature_set.subject)
    except TypeError as exc:
        raise ValueError("subject identifiers must be hashable.") from exc
    features = np.asarray(feature_set.features, dtype=float)
    if features.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix.")
    if features.shape[0] == 0:
        raise ValueError("features must contain at least one row.")
    labels = np.asarray(feature_set.labels).ravel()
    if labels.shape[0] != features.shape[0]:
        raise ValueError(f"labels length must match feature rows: {labels.shape[0]} != {features.shape[0]}.")
    trial_ids = None if feature_set.trial_ids is None else np.asarray(feature_set.trial_ids).ravel()
    if trial_ids is not None and trial_ids.shape[0] != features.shape[0]:
        raise ValueError(f"trial_ids length must match feature rows: {trial_ids.shape[0]} != {features.shape[0]}.")
    return SubjectFeatureSet(
        subject=feature_set.subject,
        features=features,
        labels=labels,
        trial_ids=trial_ids,
        metadata=feature_set.metadata,
    )


def _validate_candidates(candidates: Sequence[CrossSubjectCandidate]) -> tuple[CrossSubjectCandidate, ...]:
    validated = tuple(_validate_candidate(candidate) for candidate in candidates)
    if not validated:
        raise ValueError("At least one cross-subject candidate is required.")
    return validated


def _validate_candidate(candidate: CrossSubjectCandidate) -> CrossSubjectCandidate:
    if not isinstance(candidate, CrossSubjectCandidate):
        raise TypeError("candidates must contain CrossSubjectCandidate instances.")
    if not str(candidate.name).strip():
        raise ValueError("candidate.name must be non-empty.")
    if not callable(candidate.fit_model):
        raise ValueError("candidate.fit_model must be callable.")
    if candidate.metadata is not None and not isinstance(candidate.metadata, Mapping):
        raise TypeError("candidate.metadata must be a mapping or None.")
    return candidate


def _resolve_chance_classes(feature_sets: Sequence[SubjectFeatureSet], chance_classes: int | None) -> int:
    if chance_classes is not None:
        chance_classes = int(chance_classes)
        if chance_classes <= 0:
            raise ValueError("chance_classes must be positive.")
        return chance_classes
    labels = np.concatenate([np.asarray(feature_set.labels).ravel() for feature_set in feature_sets])
    n_classes = int(np.unique(labels).shape[0])
    if n_classes <= 0:
        raise ValueError("At least one class label is required.")
    return n_classes


def _normalize_outer_subjects(feature_sets: Sequence[SubjectFeatureSet], outer_subjects: Sequence[Hashable] | None) -> tuple[Hashable, ...]:
    subjects = tuple(feature_set.subject for feature_set in feature_sets)
    if outer_subjects is None:
        return subjects
    requested = tuple(outer_subjects)
    if not requested:
        raise ValueError("At least one outer subject is required.")
    unknown = [subject for subject in requested if subject not in subjects]
    if unknown:
        raise ValueError(f"Outer subjects must be present in feature_sets: {unknown}")
    return requested


def _normalize_selection_metric(selection_metric: str) -> str:
    normalized = str(selection_metric).strip().lower().replace("-", "_")
    if normalized not in {"accuracy", "balanced_accuracy"}:
        raise ValueError("selection_metric must be 'accuracy' or 'balanced_accuracy'.")
    return normalized


def _training_labels(
    feature_set: SubjectFeatureSet,
    *,
    label_shuffle_seed: int | None,
    label_shuffle_context: Sequence[Hashable],
) -> np.ndarray:
    labels = np.asarray(feature_set.labels).ravel()
    if label_shuffle_seed is None:
        return labels
    seed_values = [int(label_shuffle_seed), *[_stable_seed_component(value) for value in label_shuffle_context], _stable_seed_component(feature_set.subject)]
    rng = np.random.default_rng(np.random.SeedSequence(seed_values))
    return rng.permutation(labels)


def _stable_seed_component(value: Hashable) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value) & 0xFFFFFFFF
    digest = hashlib.blake2b(repr(_row_value(value)).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0xFFFFFFFF


def _predict_with_class_scores(model_bundle: WindowedModelBundle, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transformed = transform_window_features(model_bundle, features)
    model = model_bundle.model
    predictions = np.asarray(model.predict(transformed))
    classes = np.asarray(getattr(model, "classes_", np.unique(np.concatenate([model_bundle.train_labels, predictions]))))
    scores: np.ndarray | None = None
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(transformed), dtype=float)
        if scores.ndim == 1:
            if classes.shape[0] == 2:
                scores = np.column_stack((-scores, scores))
            else:
                scores = scores[:, None]
    elif hasattr(model, "predict_proba"):
        scores = np.asarray(model.predict_proba(transformed), dtype=float)
    if scores is None or scores.ndim != 2 or scores.shape[1] != classes.shape[0]:
        scores = _one_hot_prediction_scores(predictions, classes)
    return predictions, scores, classes


def _one_hot_prediction_scores(predictions: np.ndarray, classes: np.ndarray) -> np.ndarray:
    scores = np.zeros((predictions.shape[0], classes.shape[0]), dtype=float)
    class_to_column = {_row_value(class_label): class_index for class_index, class_label in enumerate(classes.tolist())}
    for row_index, prediction in enumerate(predictions.tolist()):
        column = class_to_column.get(_row_value(prediction))
        if column is not None:
            scores[row_index, column] = 1.0
    return scores


def _ranked_label_metrics(true_labels: np.ndarray, class_scores: np.ndarray, score_classes: np.ndarray) -> dict[str, Any]:
    true_labels = np.asarray(true_labels).ravel()
    class_scores = np.asarray(class_scores, dtype=float)
    score_classes = np.asarray(score_classes)
    if true_labels.size == 0 or class_scores.ndim != 2 or class_scores.shape[1] == 0:
        ranks = np.full(true_labels.shape[0], np.nan, dtype=float)
        return {
            "true_label_ranks": ranks,
            "top2_accuracy": np.nan,
            "top3_accuracy": np.nan,
            "mean_true_label_rank": np.nan,
            "median_true_label_rank": np.nan,
        }
    class_to_column = {_row_value(class_label): class_index for class_index, class_label in enumerate(score_classes.tolist())}
    ranks = np.empty(true_labels.shape[0], dtype=float)
    for row_index, label in enumerate(true_labels.tolist()):
        column = class_to_column.get(_row_value(label))
        if column is None or not np.isfinite(class_scores[row_index, column]):
            ranks[row_index] = np.inf
            continue
        true_score = class_scores[row_index, column]
        ranks[row_index] = 1.0 + float(np.sum(class_scores[row_index] > true_score))
    finite_ranks = ranks[np.isfinite(ranks)]
    return {
        "true_label_ranks": ranks,
        "top2_accuracy": float(np.mean(ranks <= 2)),
        "top3_accuracy": float(np.mean(ranks <= 3)),
        "mean_true_label_rank": float(np.mean(finite_ranks)) if finite_ranks.size else np.nan,
        "median_true_label_rank": float(np.median(finite_ranks)) if finite_ranks.size else np.nan,
    }


def _select_nested_candidate(
    inner_rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[CrossSubjectCandidate],
    *,
    selection_metric: str,
) -> dict[str, Any]:
    if not inner_rows:
        raise ValueError("At least one inner-validation row is required for nested selection.")
    candidate_indices = sorted({int(row["candidate_index"]) for row in inner_rows})
    summaries: list[dict[str, Any]] = []
    for candidate_index in candidate_indices:
        candidate_rows = [row for row in inner_rows if int(row["candidate_index"]) == candidate_index]
        metrics = _row_float_values(candidate_rows, selection_metric)
        accuracy = _row_float_values(candidate_rows, "accuracy")
        balanced = _row_float_values(candidate_rows, "balanced_accuracy")
        example = candidate_rows[0]
        candidate = candidates[candidate_index - 1]
        row: dict[str, Any] = {
            "selection_mode": "nested_loso",
            "selection_metric": selection_metric,
            "outer_fold": example["outer_fold"],
            "test_subject": example["outer_test_subject"],
            "selected_candidate_index": int(candidate_index),
            "n_candidates": len(candidates),
            "n_inner_folds": len(candidate_rows),
            "selected_inner_score_mean": float(np.mean(metrics)),
            "selected_inner_score_median": float(np.median(metrics)),
            "selected_inner_score_sem": _sem(metrics),
            "selected_inner_accuracy_mean": float(np.mean(accuracy)),
            "selected_inner_accuracy_median": float(np.median(accuracy)),
            "selected_inner_accuracy_sem": _sem(accuracy),
            "selected_inner_balanced_accuracy_mean": float(np.mean(balanced)),
            "selected_inner_balanced_accuracy_median": float(np.median(balanced)),
            "selected_inner_balanced_accuracy_sem": _sem(balanced),
        }
        _add_selected_candidate_fields(row, candidate)
        summaries.append(row)

    ranked = sorted(
        summaries,
        key=lambda row: (
            -float(row["selected_inner_score_mean"]),
            -float(row["selected_inner_score_median"]),
            int(row["selected_candidate_index"]),
        ),
    )
    selected = dict(ranked[0])
    selected["selected_inner_rank"] = 1
    if len(ranked) > 1:
        selected["selected_inner_second_best_score_mean"] = float(ranked[1]["selected_inner_score_mean"])
        selected["selected_inner_winner_margin"] = float(selected["selected_inner_score_mean"]) - float(ranked[1]["selected_inner_score_mean"])
    else:
        selected["selected_inner_second_best_score_mean"] = np.nan
        selected["selected_inner_winner_margin"] = np.nan
    return selected


def _add_candidate_fields(row: dict[str, Any], candidate: CrossSubjectCandidate, candidate_index: int) -> None:
    row["candidate_index"] = int(candidate_index)
    row["candidate_name"] = candidate.name
    row["components_pca"] = candidate.components_pca
    if candidate.train_window is not None:
        row["train_window_start"] = float(candidate.train_window[0])
        row["train_window_stop"] = float(candidate.train_window[1])
    for key, value in (candidate.metadata or {}).items():
        target_key = key if key not in row else f"candidate_{key}"
        row[target_key] = _row_value(value)


def _add_selected_candidate_fields(row: dict[str, Any], candidate: CrossSubjectCandidate) -> None:
    row["selected_candidate_name"] = candidate.name
    row["selected_components_pca"] = candidate.components_pca
    if candidate.train_window is not None:
        row["selected_train_window_start"] = float(candidate.train_window[0])
        row["selected_train_window_stop"] = float(candidate.train_window[1])
    for key, value in (candidate.metadata or {}).items():
        row[f"selected_{key}"] = _row_value(value)


def _add_selected_fields(row: dict[str, Any], selected_row: Mapping[str, Any]) -> None:
    for key, value in selected_row.items():
        row[key] = value


def _trial_ids(feature_set: SubjectFeatureSet) -> np.ndarray:
    if feature_set.trial_ids is None:
        return np.arange(np.asarray(feature_set.labels).shape[0], dtype=int)
    return np.asarray(feature_set.trial_ids).ravel()


def _row_float_values(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    try:
        return np.asarray([float(row[key]) for row in rows], dtype=float)
    except KeyError as exc:
        raise ValueError(f"Rows are missing required metric column: {key}") from exc


def _row_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _format_subjects(subjects: Sequence[Hashable]) -> str:
    return ",".join(str(_row_value(subject)) for subject in subjects)


def _sem(values: Sequence[float] | np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size <= 1:
        return np.nan
    return float(np.std(finite, ddof=1) / np.sqrt(finite.size))


def _sem_or_nan(values: Sequence[float] | np.ndarray) -> float:
    return _sem(values)


def _nanmean_or_nan(values: Sequence[float] | np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if not np.any(np.isfinite(values)):
        return np.nan
    return float(np.nanmean(values))


def _percent_nanmean_or_nan(values: Sequence[float] | np.ndarray) -> float:
    mean = _nanmean_or_nan(values)
    return np.nan if not np.isfinite(mean) else float(100.0 * mean)


def _one_sided_exact_sign_p_value(differences: Sequence[float] | np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    finite = differences[np.isfinite(differences)]
    if finite.size == 0:
        return np.nan
    successes = int(np.sum(finite > 0.0))
    n = int(finite.size)
    return float(sum(comb(n, k) for k in range(successes, n + 1)) / (2**n))
