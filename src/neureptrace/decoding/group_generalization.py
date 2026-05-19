"""Group-held-out feature-matrix benchmarks.

This module keeps the cross-subject/cross-session benchmarking machinery in the
NeuRepTrace layer without assuming anything about dataset-specific file formats.
Callers provide an already extracted feature matrix, labels, and grouping vector
(for example subject or session IDs).  The helper performs a nested
leave-one-group-out candidate selection pass and evaluates the selected decoder
on each untouched outer group exactly once.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from neureptrace.decoding import (
    make_decoder,
    make_tuning_cross_validator,
    normalize_decoder_name,
    normalize_emission_mode,
    normalize_feature_preprocessor,
    predict_emission_probabilities,
)

GROUP_GENERALIZATION_SELECTION_METRICS = ("accuracy", "balanced_accuracy")
DEFAULT_GROUP_GENERALIZATION_SELECTION_METRIC = "balanced_accuracy"


@dataclass(frozen=True)
class DecoderCandidate:
    """One decoder/preprocessor candidate for nested group-held-out selection."""

    name: str
    decoder: str = "logistic"
    emission_mode: str = "calibrated"
    feature_preprocessor: str = "none"
    pca_components: int | float | str | None = None
    classifier_param: Any = None
    tune_hyperparameters: bool = False
    tuning_cv_splits: int = 3
    tuning_scoring: str = "accuracy"
    tuning_c_grid: Sequence[float] | str | None = None
    max_iter: int = 1000
    random_state: int | None = 13
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GroupGeneralizationResult:
    """Tables emitted by a nested group-held-out benchmark."""

    outer: pd.DataFrame
    inner_validation: pd.DataFrame
    selected: pd.DataFrame
    predictions: pd.DataFrame


def make_decoder_candidate_grid(
    *,
    decoders: Sequence[str] | str = ("logistic",),
    emission_modes: Sequence[str] | str = ("calibrated",),
    feature_preprocessors: Sequence[str | None] | str | None = ("none",),
    pca_components: Sequence[int | float | str | None] | int | float | str | None = (None,),
    classifier_params: Sequence[Any] | Any = (None,),
    tune_hyperparameters: bool = False,
    tuning_cv_splits: int = 3,
    tuning_scorings: Sequence[str] | str = ("accuracy",),
    tuning_c_grids: Sequence[Sequence[float] | str | None] | Sequence[float] | str | None = (None,),
    max_iter: int = 1000,
    random_state: int | None = 13,
    name_prefix: str | None = None,
) -> tuple[DecoderCandidate, ...]:
    """Return the cartesian product of neutral decoder/preprocessor candidates."""

    candidates: list[DecoderCandidate] = []
    for decoder, emission_mode, feature_preprocessor, components, classifier_param, tuning_scoring, tuning_c_grid in product(
        _as_tuple(decoders),
        _as_tuple(emission_modes),
        _as_tuple(feature_preprocessors),
        _as_tuple(pca_components),
        _as_tuple(classifier_params),
        _as_tuple(tuning_scorings),
        _as_tuple(tuning_c_grids),
    ):
        normalized_decoder = normalize_decoder_name(str(decoder))
        normalized_emission_mode = normalize_emission_mode(str(emission_mode))
        normalized_preprocessor = normalize_feature_preprocessor(feature_preprocessor)
        name = _candidate_name(
            normalized_decoder,
            normalized_emission_mode,
            normalized_preprocessor,
            components,
            classifier_param,
            tune_hyperparameters=tune_hyperparameters,
            prefix=name_prefix,
        )
        candidates.append(
            DecoderCandidate(
                name=name,
                decoder=normalized_decoder,
                emission_mode=normalized_emission_mode,
                feature_preprocessor=normalized_preprocessor,
                pca_components=components,
                classifier_param=classifier_param,
                tune_hyperparameters=bool(tune_hyperparameters),
                tuning_cv_splits=int(tuning_cv_splits),
                tuning_scoring=str(tuning_scoring),
                tuning_c_grid=tuning_c_grid,
                max_iter=int(max_iter),
                random_state=random_state,
            )
        )
    return tuple(candidates)


def run_group_generalization_benchmark(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[Any] | np.ndarray,
    groups: Sequence[Any] | np.ndarray,
    candidates: Sequence[DecoderCandidate | str],
    *,
    outer_groups: Sequence[Any] | np.ndarray | None = None,
    selection_metric: str = DEFAULT_GROUP_GENERALIZATION_SELECTION_METRIC,
    include_predictions: bool = True,
    sample_ids: Sequence[Any] | np.ndarray | None = None,
    progress: Callable[[str], None] | None = None,
) -> GroupGeneralizationResult:
    """Run nested leave-one-group-out candidate selection and outer scoring.

    Parameters
    ----------
    features, labels, groups:
        Row-aligned feature matrix, class labels, and grouping variable.  The
        grouping variable is usually a subject, session, or acquisition site.
    candidates:
        Decoder candidates to evaluate.  Strings are accepted as a shorthand for
        ``DecoderCandidate(name=<decoder>, decoder=<decoder>)``.
    outer_groups:
        Optional subset of groups to score as untouched outer folds.  Candidate
        selection always uses only the remaining groups for each outer fold.
    selection_metric:
        Metric used to select the best candidate from the inner folds.
    include_predictions:
        Whether to emit one held-out prediction row per outer-fold sample.
    sample_ids:
        Optional row identifiers copied into the prediction table.  Defaults to
        the input row index.
    progress:
        Optional callback receiving compact status strings.
    """

    features, labels, groups, sample_ids = _validate_feature_inputs(features, labels, groups, sample_ids)
    candidates = _normalize_candidates(candidates)
    selection_metric = normalize_group_generalization_metric(selection_metric)
    outer_groups = _resolve_outer_groups(groups, outer_groups)
    if len(np.unique(groups)) < 3:
        raise ValueError("At least three groups are required for nested group-held-out benchmarking.")

    outer_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for outer_group_index, outer_group in enumerate(outer_groups):
        if progress is not None:
            progress(f"START outer_group={outer_group}")
        train_mask = groups != outer_group
        test_mask = groups == outer_group
        _require_nonempty_fold(test_mask, f"outer group {outer_group!r}")
        _require_two_classes(labels[train_mask], f"outer training fold for group {outer_group!r}")

        selection_summaries = []
        for candidate_index, candidate in enumerate(candidates):
            candidate_inner_rows = _evaluate_inner_group_folds(
                features,
                labels,
                groups,
                train_mask,
                outer_group=outer_group,
                candidate_index=candidate_index,
                candidate=candidate,
            )
            inner_rows.extend(candidate_inner_rows)
            selection_summaries.append(_selection_summary(candidate_inner_rows, selection_metric, candidate_index, candidate))

        selected_summary = _select_candidate(selection_summaries, selection_metric)
        selected_candidate = candidates[int(selected_summary["candidate_index"])]
        selected_rows.append({"outer_group": outer_group, **selected_summary})

        model = _fit_candidate(
            selected_candidate,
            features[train_mask],
            labels[train_mask],
            groups=groups[train_mask],
        )
        outer_metrics = _score_model(
            model,
            features[test_mask],
            labels[test_mask],
            train_labels=labels[train_mask],
        )
        outer_row = {
            "outer_group_index": outer_group_index,
            "outer_group": outer_group,
            "n_train_groups": int(np.unique(groups[train_mask]).size),
            "n_train_samples": int(np.sum(train_mask)),
            "n_test_samples": int(np.sum(test_mask)),
            **_candidate_row(int(selected_summary["candidate_index"]), selected_candidate),
            f"selected_inner_{selection_metric}_mean": selected_summary[f"inner_{selection_metric}_mean"],
            f"selected_inner_{selection_metric}_std": selected_summary[f"inner_{selection_metric}_std"],
            f"selected_inner_{selection_metric}_sem": selected_summary[f"inner_{selection_metric}_sem"],
            "selected_inner_n_folds": selected_summary["inner_n_folds"],
            **outer_metrics,
        }
        outer_rows.append(outer_row)

        if include_predictions:
            prediction_rows.extend(
                _prediction_rows(
                    model,
                    features[test_mask],
                    labels[test_mask],
                    groups[test_mask],
                    sample_ids[test_mask],
                    outer_group=outer_group,
                    candidate_index=int(selected_summary["candidate_index"]),
                    candidate=selected_candidate,
                )
            )
        if progress is not None:
            progress(f"DONE outer_group={outer_group} {selection_metric}={outer_metrics[selection_metric]:.4f}")

    return GroupGeneralizationResult(
        outer=pd.DataFrame(outer_rows),
        inner_validation=pd.DataFrame(inner_rows),
        selected=pd.DataFrame(selected_rows),
        predictions=pd.DataFrame(prediction_rows),
    )


def summarize_group_generalization(result: GroupGeneralizationResult | pd.DataFrame, *, metric: str = DEFAULT_GROUP_GENERALIZATION_SELECTION_METRIC) -> pd.DataFrame:
    """Summarize outer-fold scores across held-out groups."""

    metric = normalize_group_generalization_metric(metric)
    outer = result.outer if isinstance(result, GroupGeneralizationResult) else result
    if metric not in outer.columns:
        raise ValueError(f"Outer result table does not contain metric column {metric!r}.")
    values = pd.to_numeric(outer[metric], errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        mean = std = sem = median = np.nan
    elif finite.size == 1:
        mean = median = float(finite[0])
        std = sem = np.nan
    else:
        mean = float(np.mean(finite))
        median = float(np.median(finite))
        std = float(np.std(finite, ddof=1))
        sem = float(std / np.sqrt(finite.size))
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "n_outer_folds": int(len(outer)),
                "n_finite_outer_folds": int(finite.size),
                f"{metric}_mean": mean,
                f"{metric}_std": std,
                f"{metric}_sem": sem,
                f"{metric}_median": median,
            }
        ]
    )


def normalize_group_generalization_metric(metric: str) -> str:
    """Normalize metric aliases used for group-generalization selection."""

    normalized = str(metric).strip().lower().replace("-", "_")
    if normalized in {"acc", "accuracy"}:
        return "accuracy"
    if normalized in {"balanced", "balanced_acc", "balanced_accuracy", "bal_acc"}:
        return "balanced_accuracy"
    raise ValueError(f"Unknown group-generalization metric {metric!r}. Available metrics: {', '.join(GROUP_GENERALIZATION_SELECTION_METRICS)}.")


def _evaluate_inner_group_folds(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    outer_train_mask: np.ndarray,
    *,
    outer_group: Any,
    candidate_index: int,
    candidate: DecoderCandidate,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    inner_groups = np.unique(groups[outer_train_mask])
    if inner_groups.size < 2:
        raise ValueError("At least two inner-training groups are required after holding out the outer group.")
    for inner_group_index, inner_group in enumerate(inner_groups):
        inner_train_mask = outer_train_mask & (groups != inner_group)
        inner_test_mask = groups == inner_group
        _require_two_classes(labels[inner_train_mask], f"inner training fold for group {inner_group!r}")
        model = _fit_candidate(candidate, features[inner_train_mask], labels[inner_train_mask], groups=groups[inner_train_mask])
        rows.append(
            {
                "outer_group": outer_group,
                "inner_group_index": int(inner_group_index),
                "inner_group": inner_group,
                "n_train_groups": int(np.unique(groups[inner_train_mask]).size),
                "n_train_samples": int(np.sum(inner_train_mask)),
                "n_validation_samples": int(np.sum(inner_test_mask)),
                **_candidate_row(candidate_index, candidate),
                **_score_model(model, features[inner_test_mask], labels[inner_test_mask], train_labels=labels[inner_train_mask]),
            }
        )
    return rows


def _fit_candidate(candidate: DecoderCandidate, features: np.ndarray, labels: np.ndarray, *, groups: np.ndarray | None):
    tuning_cv: int | Sequence[tuple[np.ndarray, np.ndarray]] = candidate.tuning_cv_splits
    if candidate.tune_hyperparameters:
        tuning_cv = make_tuning_cross_validator(labels, groups, int(candidate.tuning_cv_splits))
    model = make_decoder(
        candidate.decoder,
        max_iter=candidate.max_iter,
        emission_mode=candidate.emission_mode,
        feature_preprocessor=candidate.feature_preprocessor,
        pca_components=candidate.pca_components,
        tune_hyperparameters=candidate.tune_hyperparameters,
        tuning_cv=tuning_cv,
        tuning_scoring=candidate.tuning_scoring,
        tuning_c_grid=candidate.tuning_c_grid,
        classifier_param=candidate.classifier_param,
        random_state=candidate.random_state,
    )
    model.fit(features, labels)
    return model


def _score_model(model, features: np.ndarray, labels: np.ndarray, *, train_labels: np.ndarray) -> dict[str, Any]:
    predictions = np.asarray(model.predict(features))
    train_classes = np.unique(train_labels)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "chance_accuracy": float(1.0 / train_classes.size) if train_classes.size else np.nan,
        "n_classes_train": int(train_classes.size),
        "n_classes_test": int(np.unique(labels).size),
    }


def _prediction_rows(
    model,
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    sample_ids: np.ndarray,
    *,
    outer_group: Any,
    candidate_index: int,
    candidate: DecoderCandidate,
) -> list[dict[str, Any]]:
    predictions = np.asarray(model.predict(features))
    probabilities, score_classes = _safe_probabilities(model, features, emission_mode=candidate.emission_mode)
    candidate_fields = _candidate_row(candidate_index, candidate)
    rows: list[dict[str, Any]] = []
    for row_index, (sample_id, group, true_label, predicted_label) in enumerate(zip(sample_ids, groups, labels, predictions, strict=True)):
        probability_fields = _row_probability_fields(probabilities, score_classes, row_index, true_label)
        rows.append(
            {
                "outer_group": outer_group,
                "test_group": group,
                "sample_id": sample_id,
                **candidate_fields,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "correct": bool(predicted_label == true_label),
                **probability_fields,
            }
        )
    return rows


def _safe_probabilities(model, features: np.ndarray, *, emission_mode: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        probabilities = predict_emission_probabilities(model, features, emission_mode=emission_mode)
    except (AttributeError, ValueError):
        return None, None
    classes = getattr(model, "classes_", None)
    if classes is None:
        return probabilities, None
    return probabilities, np.asarray(classes)


def _row_probability_fields(probabilities: np.ndarray | None, score_classes: np.ndarray | None, row_index: int, true_label: Any) -> dict[str, Any]:
    if probabilities is None or probabilities.ndim != 2:
        return {
            "predicted_score": np.nan,
            "true_label_score": np.nan,
            "true_label_rank": np.nan,
            "top2_correct": False,
            "top3_correct": False,
        }
    scores = probabilities[row_index]
    predicted_score = float(np.max(scores)) if scores.size else np.nan
    if score_classes is None:
        return {
            "predicted_score": predicted_score,
            "true_label_score": np.nan,
            "true_label_rank": np.nan,
            "top2_correct": False,
            "top3_correct": False,
        }
    matches = np.flatnonzero(score_classes == true_label)
    if matches.size == 0:
        true_label_score = np.nan
        true_label_rank = np.nan
    else:
        true_label_score = float(scores[int(matches[0])])
        true_label_rank = float(1 + np.sum(scores > true_label_score))
    return {
        "predicted_score": predicted_score,
        "true_label_score": true_label_score,
        "true_label_rank": true_label_rank,
        "top2_correct": bool(np.isfinite(true_label_rank) and true_label_rank <= 2),
        "top3_correct": bool(np.isfinite(true_label_rank) and true_label_rank <= 3),
    }


def _selection_summary(rows: Sequence[dict[str, Any]], metric: str, candidate_index: int, candidate: DecoderCandidate) -> dict[str, Any]:
    values = np.asarray([row.get(metric, np.nan) for row in rows], dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        mean = std = sem = median = np.nan
    elif finite.size == 1:
        mean = median = float(finite[0])
        std = sem = np.nan
    else:
        mean = float(np.mean(finite))
        median = float(np.median(finite))
        std = float(np.std(finite, ddof=1))
        sem = float(std / np.sqrt(finite.size))
    return {
        **_candidate_row(candidate_index, candidate),
        "inner_n_folds": int(len(rows)),
        "inner_n_finite_folds": int(finite.size),
        f"inner_{metric}_mean": mean,
        f"inner_{metric}_std": std,
        f"inner_{metric}_sem": sem,
        f"inner_{metric}_median": median,
    }


def _select_candidate(summaries: Sequence[dict[str, Any]], metric: str) -> dict[str, Any]:
    if not summaries:
        raise ValueError("At least one candidate summary is required.")
    metric_column = f"inner_{metric}_mean"
    ranked = sorted(
        summaries,
        key=lambda row: (_finite_or_negative_infinity(row[metric_column]), -int(row["candidate_index"])),
        reverse=True,
    )
    if not np.isfinite(ranked[0][metric_column]):
        raise ValueError(f"No candidate has a finite inner-validation {metric} score.")
    return dict(ranked[0])


def _candidate_row(candidate_index: int, candidate: DecoderCandidate) -> dict[str, Any]:
    row = {
        "candidate_index": int(candidate_index),
        "candidate_name": candidate.name,
        "decoder": candidate.decoder,
        "emission_mode": candidate.emission_mode,
        "feature_preprocessor": candidate.feature_preprocessor,
        "pca_components": _scalar_or_empty(candidate.pca_components),
        "classifier_param": _scalar_or_empty(candidate.classifier_param),
        "tune_hyperparameters": bool(candidate.tune_hyperparameters),
        "tuning_cv_splits": int(candidate.tuning_cv_splits),
        "tuning_scoring": candidate.tuning_scoring,
        "tuning_c_grid": _scalar_or_empty(candidate.tuning_c_grid),
        "max_iter": int(candidate.max_iter),
        "random_state": _scalar_or_empty(candidate.random_state),
    }
    for key, value in candidate.metadata.items():
        row[f"candidate_{key}"] = value
    return row


def _normalize_candidates(candidates: Sequence[DecoderCandidate | str]) -> tuple[DecoderCandidate, ...]:
    normalized: list[DecoderCandidate] = []
    for candidate in candidates:
        if isinstance(candidate, DecoderCandidate):
            normalized.append(candidate)
        elif isinstance(candidate, str):
            decoder = normalize_decoder_name(candidate)
            normalized.append(DecoderCandidate(name=decoder, decoder=decoder))
        else:
            raise TypeError("candidates must contain DecoderCandidate instances or decoder-name strings.")
    if not normalized:
        raise ValueError("At least one decoder candidate is required.")
    return tuple(normalized)


def _validate_feature_inputs(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[Any] | np.ndarray,
    groups: Sequence[Any] | np.ndarray,
    sample_ids: Sequence[Any] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels).ravel()
    groups = np.asarray(groups).ravel()
    if features.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix.")
    if labels.shape[0] != features.shape[0]:
        raise ValueError("labels must contain one value per feature row.")
    if groups.shape[0] != features.shape[0]:
        raise ValueError("groups must contain one value per feature row.")
    if sample_ids is None:
        sample_ids = np.arange(features.shape[0], dtype=int)
    sample_ids = np.asarray(sample_ids).ravel()
    if sample_ids.shape[0] != features.shape[0]:
        raise ValueError("sample_ids must contain one value per feature row.")
    return features, labels, groups, sample_ids


def _resolve_outer_groups(groups: np.ndarray, outer_groups: Sequence[Any] | np.ndarray | None) -> np.ndarray:
    unique_groups = np.unique(groups)
    if outer_groups is None:
        return unique_groups
    requested = np.asarray(outer_groups).ravel()
    if requested.size == 0:
        raise ValueError("outer_groups must contain at least one group when provided.")
    missing = [group for group in requested if not np.any(unique_groups == group)]
    if missing:
        raise ValueError(f"outer_groups contain values not present in groups: {missing}")
    return requested


def _require_nonempty_fold(mask: np.ndarray, name: str) -> None:
    if not np.any(mask):
        raise ValueError(f"{name} does not contain any samples.")


def _require_two_classes(labels: np.ndarray, name: str) -> None:
    if np.unique(labels).size < 2:
        raise ValueError(f"Need at least two classes in {name}.")


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None or isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, np.ndarray):
        return tuple(value.tolist())
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _candidate_name(
    decoder: str,
    emission_mode: str,
    feature_preprocessor: str,
    pca_components: Any,
    classifier_param: Any,
    *,
    tune_hyperparameters: bool,
    prefix: str | None,
) -> str:
    parts = [] if prefix is None else [str(prefix)]
    parts.extend([decoder, emission_mode, feature_preprocessor])
    if pca_components is not None:
        parts.append(f"pca={pca_components}")
    if classifier_param is not None:
        parts.append(f"param={classifier_param}")
    if tune_hyperparameters:
        parts.append("tuned")
    return "__".join(parts)


def _scalar_or_empty(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return value


def _finite_or_negative_infinity(value: Any) -> float:
    value = float(value)
    return value if np.isfinite(value) else float("-inf")
