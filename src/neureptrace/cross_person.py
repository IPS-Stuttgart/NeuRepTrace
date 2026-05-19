"""Nested cross-person decoding for time-aligned M/EEG representations.

This module provides a dataset-independent engine for testing whether a decoder
trained on source subjects generalizes to a held-out target subject.  It keeps
all target-task labels out of preprocessing, model selection, classifier fitting,
and probability calibration.  Optional target-side adaptation is explicit:
``cue_class_procrustes`` may use an independent cue/localizer matrix attached to
the target subject, while ``train_class_procrustes`` is a source-derived control.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss

from neureptrace.decoding import make_decoder, predict_emission_probabilities
from neureptrace.observations import stable_hash

DEFAULT_BASELINE_WINDOW = (-0.35, -0.05)
DEFAULT_WINDOW_CENTERS = (0.150, 0.175, 0.200)
DEFAULT_WINDOW_SIZE = 0.100
FEATURE_MODES = ("sensor_mean", "sensor_flat", "sensor_mean_slope", "sensor_mean_slope_std")
NORMALIZATION_MODES = ("none", "subject_z", "subject_trial_z", "subject_baseline_z", "subject_baseline_whiten")
ALIGNMENT_MODES = ("none", "train_class_procrustes", "source_class_procrustes", "cue_class_procrustes")
SELECTION_DIVERSITY_MODES = ("none", "window", "alignment", "decoder")


@dataclass(frozen=True)
class CrossPersonCandidate:
    """One candidate decoding configuration evaluated inside nested LOSO."""

    window_center: float = DEFAULT_WINDOW_CENTERS[1]
    window_size: float = DEFAULT_WINDOW_SIZE
    baseline_window: tuple[float, float] = DEFAULT_BASELINE_WINDOW
    feature_mode: str = "sensor_flat"
    normalization: str = "subject_baseline_z"
    alignment: str = "none"
    decoder: str = "linear_svm"
    emission_mode: str = "calibrated"
    feature_preprocessor: str = "pca_whiten"
    pca_components: int | float | str | None = 64
    classifier_param: Any = None
    max_iter: int = 1000
    max_trials_per_class_per_subject: int | None = None
    random_state: int = 13

    def normalized(self) -> "CrossPersonCandidate":
        feature_mode = str(self.feature_mode).strip().lower().replace("-", "_")
        normalization = str(self.normalization).strip().lower().replace("-", "_")
        alignment = str(self.alignment).strip().lower().replace("-", "_")
        if feature_mode not in FEATURE_MODES:
            raise ValueError(f"feature_mode must be one of {FEATURE_MODES}; got {self.feature_mode!r}.")
        if normalization not in NORMALIZATION_MODES:
            raise ValueError(f"normalization must be one of {NORMALIZATION_MODES}; got {self.normalization!r}.")
        if alignment not in ALIGNMENT_MODES:
            raise ValueError(f"alignment must be one of {ALIGNMENT_MODES}; got {self.alignment!r}.")
        if self.window_size <= 0.0:
            raise ValueError("window_size must be positive.")
        if self.baseline_window[1] < self.baseline_window[0]:
            raise ValueError("baseline_window stop must be greater than or equal to start.")
        if self.max_trials_per_class_per_subject is not None and self.max_trials_per_class_per_subject <= 0:
            raise ValueError("max_trials_per_class_per_subject must be positive when set.")
        return replace(
            self,
            window_center=float(self.window_center),
            window_size=float(self.window_size),
            baseline_window=(float(self.baseline_window[0]), float(self.baseline_window[1])),
            feature_mode=feature_mode,
            normalization=normalization,
            alignment=alignment,
        )

    def key(self) -> str:
        payload = asdict(self.normalized())
        return stable_hash(payload)

    def compact_label(self) -> str:
        candidate = self.normalized()
        return (
            f"t={candidate.window_center:.3f};w={candidate.window_size:.3f};"
            f"feat={candidate.feature_mode};norm={candidate.normalization};"
            f"align={candidate.alignment};dec={candidate.decoder};prep={candidate.feature_preprocessor};"
            f"pca={candidate.pca_components}"
        )


@dataclass
class SubjectFeatureMatrix:
    """A subject's feature matrix for one candidate configuration."""

    subject: str
    features: np.ndarray
    labels: np.ndarray
    trial_indices: np.ndarray | None = None
    calibration_features: np.ndarray | None = None
    calibration_labels: np.ndarray | None = None
    metadata: dict[str, Any] | None = None

    def validated(self) -> "SubjectFeatureMatrix":
        features = np.asarray(self.features, dtype=float)
        labels = np.asarray(self.labels)
        if features.ndim != 2:
            raise ValueError(f"Subject {self.subject}: features must be 2D, got shape {features.shape}.")
        if labels.ndim != 1:
            labels = labels.ravel()
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Subject {self.subject}: feature rows ({features.shape[0]}) do not match labels ({labels.shape[0]})."
            )
        trial_indices = self.trial_indices
        if trial_indices is None:
            trial_indices = np.arange(labels.shape[0], dtype=int)
        else:
            trial_indices = np.asarray(trial_indices, dtype=int).ravel()
            if trial_indices.shape[0] != labels.shape[0]:
                raise ValueError(
                    f"Subject {self.subject}: trial index count ({trial_indices.shape[0]}) does not match labels ({labels.shape[0]})."
                )
        calibration_features = None if self.calibration_features is None else np.asarray(self.calibration_features, dtype=float)
        calibration_labels = None if self.calibration_labels is None else np.asarray(self.calibration_labels).ravel()
        if calibration_features is not None or calibration_labels is not None:
            if calibration_features is None or calibration_labels is None:
                raise ValueError(f"Subject {self.subject}: calibration_features and calibration_labels must be supplied together.")
            if calibration_features.ndim != 2:
                raise ValueError(f"Subject {self.subject}: calibration_features must be 2D.")
            if calibration_features.shape[0] != calibration_labels.shape[0]:
                raise ValueError(f"Subject {self.subject}: calibration feature rows do not match calibration labels.")
            if calibration_features.shape[1] != features.shape[1]:
                raise ValueError(
                    f"Subject {self.subject}: calibration feature dimension ({calibration_features.shape[1]}) "
                    f"does not match scored feature dimension ({features.shape[1]})."
                )
        return SubjectFeatureMatrix(
            subject=str(self.subject),
            features=features,
            labels=labels,
            trial_indices=trial_indices,
            calibration_features=calibration_features,
            calibration_labels=calibration_labels,
            metadata=dict(self.metadata or {}),
        )


FeatureLoader = Callable[[str, CrossPersonCandidate], SubjectFeatureMatrix]


@dataclass
class FoldPrediction:
    y_true: np.ndarray
    y_pred: np.ndarray
    probabilities: np.ndarray
    classes: np.ndarray
    trial_indices: np.ndarray
    candidate_rows: list[dict[str, Any]]
    alignment_metadata: dict[str, Any]

    @property
    def accuracy(self) -> float:
        return float(accuracy_score(self.y_true, self.y_pred))

    @property
    def balanced_accuracy(self) -> float:
        return float(balanced_accuracy_score(self.y_true, self.y_pred))

    @property
    def log_loss(self) -> float:
        try:
            return float(log_loss(self.y_true, self.probabilities, labels=self.classes))
        except ValueError:
            return float("nan")


@dataclass
class NestedCrossPersonArtifacts:
    outer: list[dict[str, Any]]
    inner_validation: list[dict[str, Any]]
    selected: list[dict[str, Any]]
    predictions: list[dict[str, Any]]

    def as_dict(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "outer": self.outer,
            "inner_validation": self.inner_validation,
            "selected": self.selected,
            "predictions": self.predictions,
        }


def make_cross_person_candidate_grid(
    *,
    window_centers: Sequence[float] = DEFAULT_WINDOW_CENTERS,
    window_size: float = DEFAULT_WINDOW_SIZE,
    baseline_window: tuple[float, float] = DEFAULT_BASELINE_WINDOW,
    feature_modes: Sequence[str] = ("sensor_flat",),
    normalizations: Sequence[str] = ("subject_baseline_z",),
    alignments: Sequence[str] = ("none",),
    decoders: Sequence[str] = ("linear_svm",),
    emission_modes: Sequence[str] = ("calibrated",),
    feature_preprocessors: Sequence[str] = ("pca_whiten",),
    pca_components_values: Sequence[int | float | str | None] = (64,),
    classifier_params: Sequence[Any] = (None,),
    max_iter: int = 1000,
    max_trials_per_class_per_subject: int | None = None,
    random_state: int = 13,
) -> tuple[CrossPersonCandidate, ...]:
    candidates: list[CrossPersonCandidate] = []
    for window_center in window_centers:
        for feature_mode in feature_modes:
            for normalization in normalizations:
                for alignment in alignments:
                    for decoder in decoders:
                        for emission_mode in emission_modes:
                            for feature_preprocessor in feature_preprocessors:
                                for pca_components in pca_components_values:
                                    for classifier_param in classifier_params:
                                        candidates.append(
                                            CrossPersonCandidate(
                                                window_center=float(window_center),
                                                window_size=float(window_size),
                                                baseline_window=baseline_window,
                                                feature_mode=feature_mode,
                                                normalization=normalization,
                                                alignment=alignment,
                                                decoder=decoder,
                                                emission_mode=emission_mode,
                                                feature_preprocessor=feature_preprocessor,
                                                pca_components=pca_components,
                                                classifier_param=classifier_param,
                                                max_iter=max_iter,
                                                max_trials_per_class_per_subject=max_trials_per_class_per_subject,
                                                random_state=random_state,
                                            ).normalized()
                                        )
    return tuple(candidates)


def run_nested_cross_person_from_loader(  # pylint: disable=too-many-arguments,too-many-locals
    subjects: Sequence[str],
    *,
    candidate_configs: Sequence[CrossPersonCandidate],
    feature_loader: FeatureLoader,
    outer_subjects: Sequence[str] | None = None,
    selection_metric: str = "balanced_accuracy",
    selection_ensemble_size: int = 1,
    selection_ensemble_diversity: str = "none",
    label_shuffle_control: bool = False,
    label_shuffle_seed: int = 0,
    target_calibration_label_shuffle_control: bool = False,
    target_calibration_label_shuffle_seed: int = 0,
    progress: Callable[[str], None] | None = None,
    allow_failed_candidates: bool = True,
) -> NestedCrossPersonArtifacts:
    """Run nested leave-one-subject-out cross-person decoding.

    Parameters
    ----------
    subjects:
        Subject identifiers available for outer LOSO.
    candidate_configs:
        Candidate decoding configurations.  Candidate selection is performed
        only with inner LOSO over outer-training subjects.
    feature_loader:
        Callable that returns a ``SubjectFeatureMatrix`` for a subject/candidate
        pair.  This lets dataset-specific adapters keep their own loaders while
        sharing the leakage-safe model-selection logic.
    """

    subjects = tuple(str(subject) for subject in subjects)
    if len(subjects) < 3:
        raise ValueError("At least three subjects are required for nested cross-person decoding.")
    candidates = tuple(candidate.normalized() for candidate in candidate_configs)
    if not candidates:
        raise ValueError("At least one candidate configuration is required.")
    if selection_metric not in {"accuracy", "balanced_accuracy", "log_loss"}:
        raise ValueError("selection_metric must be one of accuracy, balanced_accuracy, log_loss.")
    if selection_ensemble_size < 1:
        raise ValueError("selection_ensemble_size must be at least one.")
    selection_ensemble_diversity = str(selection_ensemble_diversity).strip().lower().replace("-", "_")
    if selection_ensemble_diversity not in SELECTION_DIVERSITY_MODES:
        raise ValueError(f"selection_ensemble_diversity must be one of {SELECTION_DIVERSITY_MODES}.")
    outer_subjects = subjects if outer_subjects is None else tuple(str(subject) for subject in outer_subjects)
    missing_outer = sorted(set(outer_subjects) - set(subjects))
    if missing_outer:
        raise ValueError(f"outer_subjects must be a subset of subjects; missing {missing_outer}.")

    cache: dict[tuple[str, str], SubjectFeatureMatrix] = {}

    def load(subject: str, candidate: CrossPersonCandidate) -> SubjectFeatureMatrix:
        key = (str(subject), candidate.key())
        if key not in cache:
            cache[key] = feature_loader(str(subject), candidate).validated()
        return cache[key]

    outer_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for outer_fold, target_subject in enumerate(outer_subjects):
        source_subjects = tuple(subject for subject in subjects if subject != target_subject)
        if progress is not None:
            progress(f"START outer_fold={outer_fold} target_subject={target_subject}")
        candidate_summaries: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(candidates):
            validation_scores: list[float] = []
            validation_log_losses: list[float] = []
            failed_folds = 0
            for inner_fold, validation_subject in enumerate(source_subjects):
                inner_train_subjects = tuple(subject for subject in source_subjects if subject != validation_subject)
                try:
                    fold = _predict_candidate(
                        [load(subject, candidate) for subject in inner_train_subjects],
                        load(validation_subject, candidate),
                        candidate,
                        label_shuffle_seed=_fold_seed(label_shuffle_seed, outer_fold, inner_fold, candidate_index) if label_shuffle_control else None,
                        target_calibration_label_shuffle_seed=(
                            _fold_seed(target_calibration_label_shuffle_seed, outer_fold, inner_fold, candidate_index)
                            if target_calibration_label_shuffle_control
                            else None
                        ),
                    )
                    score = _metric_value(fold, selection_metric)
                    validation_scores.append(score)
                    validation_log_losses.append(fold.log_loss)
                    row = _candidate_metadata(candidate, candidate_index)
                    row.update(
                        {
                            "outer_fold": outer_fold,
                            "outer_target_subject": target_subject,
                            "inner_fold": inner_fold,
                            "validation_subject": validation_subject,
                            "n_inner_train_subjects": len(inner_train_subjects),
                            "accuracy": fold.accuracy,
                            "balanced_accuracy": fold.balanced_accuracy,
                            "log_loss": fold.log_loss,
                            "failed": False,
                        }
                    )
                    inner_rows.append(row)
                except Exception as exc:  # pragma: no cover - surfaced in CSV and optionally re-raised
                    failed_folds += 1
                    if not allow_failed_candidates:
                        raise
                    row = _candidate_metadata(candidate, candidate_index)
                    row.update(
                        {
                            "outer_fold": outer_fold,
                            "outer_target_subject": target_subject,
                            "inner_fold": inner_fold,
                            "validation_subject": validation_subject,
                            "n_inner_train_subjects": len(inner_train_subjects),
                            "accuracy": np.nan,
                            "balanced_accuracy": np.nan,
                            "log_loss": np.nan,
                            "failed": True,
                            "failure_reason": str(exc),
                        }
                    )
                    inner_rows.append(row)
            mean_score = _safe_nanmean(validation_scores)
            summary = _candidate_metadata(candidate, candidate_index)
            summary.update(
                {
                    "outer_fold": outer_fold,
                    "outer_target_subject": target_subject,
                    "mean_accuracy_or_balanced_accuracy": mean_score,
                    "mean_log_loss": _safe_nanmean(validation_log_losses),
                    "selection_metric": selection_metric,
                    "selection_score": mean_score,
                    "n_inner_folds": len(source_subjects),
                    "n_failed_inner_folds": failed_folds,
                }
            )
            candidate_summaries.append(summary)

        selected = _select_candidates(
            candidate_summaries,
            candidates,
            selection_metric=selection_metric,
            ensemble_size=selection_ensemble_size,
            diversity=selection_ensemble_diversity,
        )
        for rank, candidate in enumerate(selected):
            row = _candidate_metadata(candidate, candidates.index(candidate))
            row.update(
                {
                    "outer_fold": outer_fold,
                    "outer_target_subject": target_subject,
                    "selection_rank": rank + 1,
                    "selection_ensemble_size": len(selected),
                }
            )
            selected_rows.append(row)

        fold_seed = _fold_seed(label_shuffle_seed, outer_fold, 0, 0) if label_shuffle_control else None
        target_calibration_seed = (
            _fold_seed(target_calibration_label_shuffle_seed, outer_fold, 0, 0)
            if target_calibration_label_shuffle_control
            else None
        )
        ensemble_prediction = _predict_candidate_ensemble(
            source_subjects,
            target_subject,
            selected,
            load,
            label_shuffle_seed=fold_seed,
            target_calibration_label_shuffle_seed=target_calibration_seed,
        )
        outer_row = {
            "outer_fold": outer_fold,
            "target_subject": target_subject,
            "n_train_subjects": len(source_subjects),
            "selection_metric": selection_metric,
            "selection_ensemble_size": len(selected),
            "selection_ensemble_diversity": selection_ensemble_diversity,
            "accuracy": ensemble_prediction.accuracy,
            "balanced_accuracy": ensemble_prediction.balanced_accuracy,
            "log_loss": ensemble_prediction.log_loss,
            "n_test_trials": int(ensemble_prediction.y_true.shape[0]),
            "n_classes": int(ensemble_prediction.classes.shape[0]),
            "selected_candidate_keys": "|".join(candidate.key() for candidate in selected),
            "selected_candidate_labels": "|".join(candidate.compact_label() for candidate in selected),
            "label_shuffle_control": bool(label_shuffle_control),
            "label_shuffle_seed": "" if not label_shuffle_control else int(label_shuffle_seed),
            "target_calibration_label_shuffle_control": bool(target_calibration_label_shuffle_control),
            "target_calibration_label_shuffle_seed": "" if not target_calibration_label_shuffle_control else int(target_calibration_label_shuffle_seed),
        }
        outer_rows.append(outer_row)
        prediction_rows.extend(
            _prediction_rows(
                ensemble_prediction,
                outer_fold=outer_fold,
                target_subject=target_subject,
                selected=selected,
            )
        )
        if progress is not None:
            progress(
                f"DONE outer_fold={outer_fold} target_subject={target_subject} "
                f"balanced_accuracy={ensemble_prediction.balanced_accuracy:.4f}"
            )

    return NestedCrossPersonArtifacts(
        outer=outer_rows,
        inner_validation=inner_rows,
        selected=selected_rows,
        predictions=prediction_rows,
    )


def _metric_value(fold: FoldPrediction, metric: str) -> float:
    if metric == "accuracy":
        return fold.accuracy
    if metric == "balanced_accuracy":
        return fold.balanced_accuracy
    if metric == "log_loss":
        value = fold.log_loss
        return -value if np.isfinite(value) else np.nan
    raise ValueError(metric)


def _safe_nanmean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    finite = np.asarray(values, dtype=float)
    if not np.isfinite(finite).any():
        return float("nan")
    return float(np.nanmean(finite))


def _fold_seed(seed: int, *context: int) -> int:
    return int(np.random.SeedSequence([int(seed), *[int(value) for value in context]]).generate_state(1)[0])


def _candidate_metadata(candidate: CrossPersonCandidate, candidate_index: int) -> dict[str, Any]:
    candidate = candidate.normalized()
    return {
        "candidate_index": int(candidate_index),
        "candidate_key": candidate.key(),
        "candidate_label": candidate.compact_label(),
        **asdict(candidate),
        "classifier_param": _jsonish(candidate.classifier_param),
        "pca_components": "" if candidate.pca_components is None else candidate.pca_components,
        "max_trials_per_class_per_subject": "" if candidate.max_trials_per_class_per_subject is None else candidate.max_trials_per_class_per_subject,
    }


def _jsonish(value: Any) -> str:
    if value is None:
        return ""
    try:
        return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    except TypeError:
        return repr(value)


def _select_candidates(
    summaries: Sequence[dict[str, Any]],
    candidates: Sequence[CrossPersonCandidate],
    *,
    selection_metric: str,
    ensemble_size: int,
    diversity: str,
) -> tuple[CrossPersonCandidate, ...]:
    del selection_metric  # the summary rows already contain selection_score
    finite = [row for row in summaries if np.isfinite(float(row.get("selection_score", np.nan)))]
    if not finite:
        raise ValueError("No candidate has a finite inner-validation score.")
    finite = sorted(finite, key=lambda row: float(row["selection_score"]), reverse=True)
    selected_rows: list[dict[str, Any]] = []
    used_values: set[Any] = set()
    diversity_field = {
        "window": "window_center",
        "alignment": "alignment",
        "decoder": "decoder",
    }.get(diversity)
    if diversity_field is not None:
        for row in finite:
            value = row[diversity_field]
            if value in used_values:
                continue
            selected_rows.append(row)
            used_values.add(value)
            if len(selected_rows) >= ensemble_size:
                break
    if len(selected_rows) < ensemble_size:
        selected_keys = {row["candidate_key"] for row in selected_rows}
        for row in finite:
            if row["candidate_key"] in selected_keys:
                continue
            selected_rows.append(row)
            selected_keys.add(row["candidate_key"])
            if len(selected_rows) >= ensemble_size:
                break
    candidate_by_key = {candidate.key(): candidate for candidate in candidates}
    return tuple(candidate_by_key[row["candidate_key"]] for row in selected_rows)


def _predict_candidate_ensemble(
    source_subjects: Sequence[str],
    target_subject: str,
    candidates: Sequence[CrossPersonCandidate],
    load: Callable[[str, CrossPersonCandidate], SubjectFeatureMatrix],
    *,
    label_shuffle_seed: int | None,
    target_calibration_label_shuffle_seed: int | None,
) -> FoldPrediction:
    predictions: list[FoldPrediction] = []
    for candidate_index, candidate in enumerate(candidates):
        predictions.append(
            _predict_candidate(
                [load(subject, candidate) for subject in source_subjects],
                load(target_subject, candidate),
                candidate,
                label_shuffle_seed=None if label_shuffle_seed is None else _fold_seed(label_shuffle_seed, candidate_index),
                target_calibration_label_shuffle_seed=(
                    None if target_calibration_label_shuffle_seed is None else _fold_seed(target_calibration_label_shuffle_seed, candidate_index)
                ),
            )
        )
    if len(predictions) == 1:
        return predictions[0]

    reference = predictions[0]
    for prediction in predictions[1:]:
        if not np.array_equal(prediction.y_true, reference.y_true) or not np.array_equal(prediction.trial_indices, reference.trial_indices):
            raise ValueError("Selected candidates produced non-aligning target trial rows; use a common trial cap/selection policy.")
    classes = np.asarray(sorted(set(np.concatenate([prediction.classes for prediction in predictions]).tolist())))
    probability_sum = np.zeros((reference.y_true.shape[0], classes.shape[0]), dtype=float)
    class_to_index = {class_value: index for index, class_value in enumerate(classes.tolist())}
    for prediction in predictions:
        for local_index, class_value in enumerate(prediction.classes.tolist()):
            probability_sum[:, class_to_index[class_value]] += prediction.probabilities[:, local_index]
    probabilities = probability_sum / float(len(predictions))
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Ensemble probability rows must have positive sums.")
    probabilities = probabilities / row_sums
    y_pred = classes[np.argmax(probabilities, axis=1)]
    return FoldPrediction(
        y_true=reference.y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        classes=classes,
        trial_indices=reference.trial_indices,
        candidate_rows=[row for prediction in predictions for row in prediction.candidate_rows],
        alignment_metadata={"ensemble_candidates": len(predictions)},
    )


def _predict_candidate(
    train_sets: Sequence[SubjectFeatureMatrix],
    target_set: SubjectFeatureMatrix,
    candidate: CrossPersonCandidate,
    *,
    label_shuffle_seed: int | None = None,
    target_calibration_label_shuffle_seed: int | None = None,
) -> FoldPrediction:
    candidate = candidate.normalized()
    train_sets = [feature_set.validated() for feature_set in train_sets]
    target_set = target_set.validated()
    if len(train_sets) < 1:
        raise ValueError("At least one source subject is required to fit a candidate.")

    aligned_train, aligned_target, alignment_metadata = _align_fold(
        train_sets,
        target_set,
        candidate,
        target_calibration_label_shuffle_seed=target_calibration_label_shuffle_seed,
    )
    train_features = np.vstack([feature_set.features for feature_set in aligned_train])
    train_labels = np.concatenate([feature_set.labels for feature_set in aligned_train])
    if label_shuffle_seed is not None:
        train_labels = _shuffle_labels_within_subjects(aligned_train, seed=label_shuffle_seed)
    if np.unique(train_labels).shape[0] < 2:
        raise ValueError("At least two classes are required after assembling source subjects.")

    model = make_decoder(
        candidate.decoder,
        max_iter=candidate.max_iter,
        emission_mode=candidate.emission_mode,
        feature_preprocessor=candidate.feature_preprocessor,
        pca_components=candidate.pca_components,
        classifier_param=candidate.classifier_param,
        random_state=candidate.random_state,
    )
    model.fit(train_features, train_labels)
    probabilities = predict_emission_probabilities(
        model,
        aligned_target.features,
        emission_mode=candidate.emission_mode,
    )
    classes = np.asarray(getattr(model, "classes_", np.unique(train_labels)))
    if probabilities.shape[1] != classes.shape[0]:
        raise ValueError(
            f"Decoder returned {probabilities.shape[1]} probability columns but exposes {classes.shape[0]} classes."
        )
    y_pred = classes[np.argmax(probabilities, axis=1)]
    candidate_row = _candidate_metadata(candidate, 0)
    candidate_row.update(
        {
            "train_subjects": "|".join(feature_set.subject for feature_set in train_sets),
            "target_subject": target_set.subject,
            "model_hash": stable_hash(
                {
                    "candidate": asdict(candidate),
                    "train_subjects": tuple(feature_set.subject for feature_set in train_sets),
                    "classes": tuple(map(str, classes.tolist())),
                    "label_shuffle": label_shuffle_seed is not None,
                    "target_calibration_label_shuffle": target_calibration_label_shuffle_seed is not None,
                }
            ),
        }
    )
    return FoldPrediction(
        y_true=aligned_target.labels,
        y_pred=y_pred,
        probabilities=probabilities,
        classes=classes,
        trial_indices=aligned_target.trial_indices if aligned_target.trial_indices is not None else np.arange(aligned_target.labels.shape[0]),
        candidate_rows=[candidate_row],
        alignment_metadata=alignment_metadata,
    )


def _shuffle_labels_within_subjects(train_sets: Sequence[SubjectFeatureMatrix], *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shuffled: list[np.ndarray] = []
    for feature_set in train_sets:
        labels = np.asarray(feature_set.labels).copy()
        rng.shuffle(labels)
        shuffled.append(labels)
    return np.concatenate(shuffled)


def _align_fold(
    train_sets: Sequence[SubjectFeatureMatrix],
    target_set: SubjectFeatureMatrix,
    candidate: CrossPersonCandidate,
    *,
    target_calibration_label_shuffle_seed: int | None = None,
) -> tuple[list[SubjectFeatureMatrix], SubjectFeatureMatrix, dict[str, Any]]:
    if candidate.alignment == "none":
        return list(train_sets), target_set, {"alignment": "none"}
    if candidate.alignment in {"train_class_procrustes", "source_class_procrustes"}:
        return _align_train_class_procrustes(train_sets, target_set, candidate)
    if candidate.alignment == "cue_class_procrustes":
        return _align_cue_class_procrustes(
            train_sets,
            target_set,
            candidate,
            target_calibration_label_shuffle_seed=target_calibration_label_shuffle_seed,
        )
    raise ValueError(f"Unsupported alignment: {candidate.alignment}")


def _align_train_class_procrustes(
    train_sets: Sequence[SubjectFeatureMatrix],
    target_set: SubjectFeatureMatrix,
    candidate: CrossPersonCandidate,
) -> tuple[list[SubjectFeatureMatrix], SubjectFeatureMatrix, dict[str, Any]]:
    common_classes = _common_classes([feature_set.labels for feature_set in train_sets])
    if common_classes.shape[0] < 2:
        return list(train_sets), target_set, {"alignment": candidate.alignment, "aligned": False, "reason": "fewer than two common classes"}
    source_patterns = [_class_patterns(feature_set.features, feature_set.labels, common_classes) for feature_set in train_sets]
    template = np.mean(np.stack(source_patterns, axis=0), axis=0)
    transforms = [_fit_orthogonal_transform(patterns, template) for patterns in source_patterns]
    aligned_train = [
        _replace_features(feature_set, _apply_orthogonal_transform(feature_set.features, transform))
        for feature_set, transform in zip(train_sets, transforms, strict=True)
    ]
    if candidate.alignment == "source_class_procrustes":
        aligned_target = target_set
        target_transform_name = "none"
    else:
        target_transform = _average_transforms(transforms)
        aligned_target = _replace_features(target_set, _apply_orthogonal_transform(target_set.features, target_transform))
        target_transform_name = "group_average_source_transform"
    return aligned_train, aligned_target, {
        "alignment": candidate.alignment,
        "aligned": True,
        "common_classes": "|".join(map(str, common_classes.tolist())),
        "target_transform": target_transform_name,
    }


def _align_cue_class_procrustes(
    train_sets: Sequence[SubjectFeatureMatrix],
    target_set: SubjectFeatureMatrix,
    candidate: CrossPersonCandidate,
    *,
    target_calibration_label_shuffle_seed: int | None = None,
) -> tuple[list[SubjectFeatureMatrix], SubjectFeatureMatrix, dict[str, Any]]:
    if target_set.calibration_features is None or target_set.calibration_labels is None:
        raise ValueError("cue_class_procrustes requires target calibration_features and calibration_labels.")
    for feature_set in train_sets:
        if feature_set.calibration_features is None or feature_set.calibration_labels is None:
            raise ValueError(f"cue_class_procrustes requires source calibration data for subject {feature_set.subject}.")
    target_calibration_labels = np.asarray(target_set.calibration_labels).copy()
    if target_calibration_label_shuffle_seed is not None:
        rng = np.random.default_rng(target_calibration_label_shuffle_seed)
        rng.shuffle(target_calibration_labels)
    common_classes = _common_classes(
        [*(feature_set.calibration_labels for feature_set in train_sets), target_calibration_labels]
    )
    if common_classes.shape[0] < 2:
        return list(train_sets), target_set, {"alignment": candidate.alignment, "aligned": False, "reason": "fewer than two common cue classes"}
    source_patterns = [
        _class_patterns(feature_set.calibration_features, feature_set.calibration_labels, common_classes)  # type: ignore[arg-type]
        for feature_set in train_sets
    ]
    template = np.mean(np.stack(source_patterns, axis=0), axis=0)
    source_transforms = [_fit_orthogonal_transform(patterns, template) for patterns in source_patterns]
    target_patterns = _class_patterns(target_set.calibration_features, target_calibration_labels, common_classes)
    target_transform = _fit_orthogonal_transform(target_patterns, template)
    aligned_train = [
        _replace_features(feature_set, _apply_orthogonal_transform(feature_set.features, transform))
        for feature_set, transform in zip(train_sets, source_transforms, strict=True)
    ]
    aligned_target = _replace_features(target_set, _apply_orthogonal_transform(target_set.features, target_transform))
    return aligned_train, aligned_target, {
        "alignment": candidate.alignment,
        "aligned": True,
        "common_classes": "|".join(map(str, common_classes.tolist())),
        "target_transform": "target_cue_labels" if target_calibration_label_shuffle_seed is None else "target_cue_labels_shuffled",
    }


def _common_classes(label_arrays: Iterable[np.ndarray]) -> np.ndarray:
    iterator = iter(label_arrays)
    try:
        common = set(np.asarray(next(iterator)).tolist())
    except StopIteration:
        return np.asarray([])
    for labels in iterator:
        common &= set(np.asarray(labels).tolist())
    return np.asarray(sorted(common))


def _class_patterns(features: np.ndarray, labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    rows = []
    for class_value in classes:
        mask = labels == class_value
        if not np.any(mask):
            raise ValueError(f"Missing class {class_value!r} while building class patterns.")
        rows.append(np.mean(features[mask], axis=0))
    return np.vstack(rows)


def _fit_orthogonal_transform(source_patterns: np.ndarray, target_patterns: np.ndarray) -> dict[str, np.ndarray]:
    source_center = np.mean(source_patterns, axis=0)
    target_center = np.mean(target_patterns, axis=0)
    source = source_patterns - source_center
    target = target_patterns - target_center
    u, _singular_values, vt = np.linalg.svd(source.T @ target, full_matrices=False)
    rotation = u @ vt
    return {"source_center": source_center, "target_center": target_center, "rotation": rotation}


def _apply_orthogonal_transform(features: np.ndarray, transform: dict[str, np.ndarray]) -> np.ndarray:
    return (features - transform["source_center"]) @ transform["rotation"] + transform["target_center"]


def _average_transforms(transforms: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    rotations = np.stack([transform["rotation"] for transform in transforms], axis=0)
    mean_rotation = np.mean(rotations, axis=0)
    u, _singular_values, vt = np.linalg.svd(mean_rotation, full_matrices=False)
    return {
        "source_center": np.mean(np.stack([transform["source_center"] for transform in transforms], axis=0), axis=0),
        "target_center": np.mean(np.stack([transform["target_center"] for transform in transforms], axis=0), axis=0),
        "rotation": u @ vt,
    }


def _replace_features(feature_set: SubjectFeatureMatrix, features: np.ndarray) -> SubjectFeatureMatrix:
    return SubjectFeatureMatrix(
        subject=feature_set.subject,
        features=features,
        labels=feature_set.labels,
        trial_indices=feature_set.trial_indices,
        calibration_features=feature_set.calibration_features,
        calibration_labels=feature_set.calibration_labels,
        metadata=feature_set.metadata,
    ).validated()


def _prediction_rows(
    prediction: FoldPrediction,
    *,
    outer_fold: int,
    target_subject: str,
    selected: Sequence[CrossPersonCandidate],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected_keys = "|".join(candidate.key() for candidate in selected)
    selected_labels = "|".join(candidate.compact_label() for candidate in selected)
    for row_index, (trial_index, true_label, predicted_label) in enumerate(
        zip(prediction.trial_indices, prediction.y_true, prediction.y_pred, strict=True)
    ):
        row = {
            "outer_fold": int(outer_fold),
            "target_subject": target_subject,
            "row_index": int(row_index),
            "trial_index": int(trial_index),
            "true_label": true_label,
            "predicted_label": predicted_label,
            "correct": bool(true_label == predicted_label),
            "selected_candidate_keys": selected_keys,
            "selected_candidate_labels": selected_labels,
        }
        for class_index, class_value in enumerate(prediction.classes.tolist()):
            row[f"class_{class_index}"] = class_value
            row[f"prob_class_{class_index}"] = float(prediction.probabilities[row_index, class_index])
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# MNE manifest loader and CLI
# ---------------------------------------------------------------------------


def load_mne_subject_features(
    epochs_path: Path,
    *,
    candidate: CrossPersonCandidate,
    subject: str,
    label_column: str,
    metadata_csv: Path | None = None,
    picks: str = "data",
    tmin: float | None = None,
    tmax: float | None = None,
    calibration_epochs_path: Path | None = None,
    calibration_metadata_csv: Path | None = None,
) -> SubjectFeatureMatrix:
    epochs, metadata = _read_epochs_and_metadata(epochs_path, metadata_csv)
    features, labels, trial_indices = _features_from_epochs(
        epochs,
        metadata,
        candidate=candidate,
        label_column=label_column,
        picks=picks,
        tmin=tmin,
        tmax=tmax,
        subject=subject,
    )
    calibration_features = None
    calibration_labels = None
    if calibration_epochs_path is not None:
        calibration_epochs, calibration_metadata = _read_epochs_and_metadata(calibration_epochs_path, calibration_metadata_csv)
        calibration_features, calibration_labels, _calibration_indices = _features_from_epochs(
            calibration_epochs,
            calibration_metadata,
            candidate=candidate,
            label_column=label_column,
            picks=picks,
            tmin=tmin,
            tmax=tmax,
            subject=subject,
        )
    return SubjectFeatureMatrix(
        subject=subject,
        features=features,
        labels=labels,
        trial_indices=trial_indices,
        calibration_features=calibration_features,
        calibration_labels=calibration_labels,
        metadata={"epochs": str(epochs_path)},
    ).validated()


def _read_epochs_and_metadata(epochs_path: Path, metadata_csv: Path | None) -> tuple[mne.Epochs, pd.DataFrame]:
    epochs = mne.read_epochs(epochs_path, preload=True, verbose="error")
    metadata = epochs.metadata.copy() if epochs.metadata is not None else None
    if metadata_csv is not None and str(metadata_csv) != "":
        metadata = pd.read_csv(metadata_csv)
    if metadata is None:
        raise ValueError(f"No metadata found for {epochs_path}; provide metadata_csv or epochs metadata.")
    if len(metadata) != len(epochs):
        raise ValueError(f"Metadata rows ({len(metadata)}) do not match epochs ({len(epochs)}) for {epochs_path}.")
    return epochs, metadata.reset_index(drop=True)


def _features_from_epochs(
    epochs: mne.Epochs,
    metadata: pd.DataFrame,
    *,
    candidate: CrossPersonCandidate,
    label_column: str,
    picks: str,
    tmin: float | None,
    tmax: float | None,
    subject: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate = candidate.normalized()
    if label_column not in metadata.columns:
        raise ValueError(f"Subject {subject}: label column {label_column!r} not found in metadata.")
    epochs = epochs.copy().pick(picks)
    if tmin is not None or tmax is not None:
        epochs.crop(tmin=tmin, tmax=tmax)
    labels = metadata[label_column].to_numpy()
    keep = pd.notna(labels)
    original_indices = np.arange(len(labels))[keep]
    labels = labels[keep]
    data = epochs[keep].get_data(copy=False)
    features = _extract_normalized_features(
        data,
        epochs.times,
        candidate=candidate,
        subject=subject,
    )
    trial_indices = _class_limited_indices(
        labels,
        max_trials_per_class=candidate.max_trials_per_class_per_subject,
        seed=candidate.random_state,
        subject=subject,
    )
    return features[trial_indices], labels[trial_indices], original_indices[trial_indices]


def _extract_normalized_features(
    data: np.ndarray,
    times: np.ndarray,
    *,
    candidate: CrossPersonCandidate,
    subject: str,
) -> np.ndarray:
    working = np.asarray(data, dtype=float)
    if candidate.normalization == "subject_baseline_whiten":
        working = _baseline_whiten_data(working, times, candidate.baseline_window, subject=subject)
    features = _extract_features(working, times, center=candidate.window_center, width=candidate.window_size, mode=candidate.feature_mode)
    if candidate.normalization == "none" or candidate.normalization == "subject_baseline_whiten":
        return features
    if candidate.normalization == "subject_trial_z":
        return _row_z(features)
    if candidate.normalization == "subject_z":
        return _column_z(features)
    if candidate.normalization == "subject_baseline_z":
        baseline_center = float(np.mean(candidate.baseline_window))
        baseline_width = float(candidate.baseline_window[1] - candidate.baseline_window[0])
        baseline_width = max(baseline_width, candidate.window_size)
        baseline_features = _extract_features(working, times, center=baseline_center, width=baseline_width, mode=candidate.feature_mode)
        mean = np.mean(baseline_features, axis=0, keepdims=True)
        std = np.std(baseline_features, axis=0, keepdims=True)
        std[std < 1e-12] = 1.0
        return (features - mean) / std
    raise ValueError(f"Unsupported normalization: {candidate.normalization}")


def _extract_features(data: np.ndarray, times: np.ndarray, *, center: float, width: float, mode: str) -> np.ndarray:
    start = float(center) - float(width) / 2.0
    stop = float(center) + float(width) / 2.0
    mask = (times >= start) & (times <= stop)
    if not np.any(mask):
        raise ValueError(f"No samples fall inside window [{start}, {stop}].")
    window = data[:, :, mask]
    if mode == "sensor_mean":
        return np.mean(window, axis=2)
    if mode == "sensor_flat":
        return window.reshape(window.shape[0], -1)
    if mode in {"sensor_mean_slope", "sensor_mean_slope_std"}:
        means = np.mean(window, axis=2)
        slopes = _window_slopes(window, times[mask])
        columns = [means, slopes]
        if mode == "sensor_mean_slope_std":
            columns.append(np.std(window, axis=2))
        return np.concatenate(columns, axis=1)
    raise ValueError(f"Unsupported feature mode: {mode}")


def _window_slopes(window: np.ndarray, window_times: np.ndarray) -> np.ndarray:
    centered = window_times - float(np.mean(window_times))
    denom = float(np.sum(centered**2))
    if denom <= 0.0:
        return np.zeros(window.shape[:2], dtype=float)
    return np.sum(window * centered[None, None, :], axis=2) / denom


def _baseline_whiten_data(data: np.ndarray, times: np.ndarray, baseline_window: tuple[float, float], *, subject: str) -> np.ndarray:
    mask = (times >= baseline_window[0]) & (times <= baseline_window[1])
    if not np.any(mask):
        raise ValueError(f"Subject {subject}: no samples in baseline window {baseline_window}.")
    baseline = data[:, :, mask].transpose(0, 2, 1).reshape(-1, data.shape[1])
    baseline = baseline - np.mean(baseline, axis=0, keepdims=True)
    cov = np.cov(baseline, rowvar=False)
    shrink = 1e-6 * float(np.trace(cov) / max(cov.shape[0], 1))
    cov = cov + np.eye(cov.shape[0]) * max(shrink, 1e-12)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-12)
    whitening = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return np.einsum("ij,tcj->tci", whitening, data.transpose(0, 2, 1)).transpose(0, 2, 1)


def _column_z(features: np.ndarray) -> np.ndarray:
    mean = np.mean(features, axis=0, keepdims=True)
    std = np.std(features, axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return (features - mean) / std


def _row_z(features: np.ndarray) -> np.ndarray:
    mean = np.mean(features, axis=1, keepdims=True)
    std = np.std(features, axis=1, keepdims=True)
    std[std < 1e-12] = 1.0
    return (features - mean) / std


def _class_limited_indices(labels: np.ndarray, *, max_trials_per_class: int | None, seed: int, subject: str) -> np.ndarray:
    labels = np.asarray(labels)
    if max_trials_per_class is None:
        return np.arange(labels.shape[0], dtype=int)
    seed_payload = [int(seed), abs(hash(str(subject))) % (2**32)]
    rng = np.random.default_rng(np.random.SeedSequence(seed_payload))
    selected: list[int] = []
    for label in sorted(set(labels.tolist())):
        class_indices = np.flatnonzero(labels == label)
        if class_indices.shape[0] > max_trials_per_class:
            class_indices = rng.choice(class_indices, size=max_trials_per_class, replace=False)
        selected.extend(int(index) for index in class_indices)
    return np.asarray(sorted(selected), dtype=int)


def make_manifest_feature_loader(
    manifest: pd.DataFrame,
    *,
    label_column: str | None,
    subject_column: str = "subject",
    epochs_column: str = "epochs",
    metadata_csv_column: str = "metadata_csv",
    calibration_epochs_column: str = "calibration_epochs",
    calibration_metadata_csv_column: str = "calibration_metadata_csv",
    picks: str = "data",
    tmin: float | None = None,
    tmax: float | None = None,
) -> tuple[tuple[str, ...], FeatureLoader]:
    if subject_column not in manifest.columns:
        raise ValueError(f"Manifest must contain subject column {subject_column!r}.")
    if epochs_column not in manifest.columns:
        raise ValueError(f"Manifest must contain epochs column {epochs_column!r}.")
    subjects = tuple(str(value) for value in manifest[subject_column].tolist())
    rows = {str(row[subject_column]): row for _index, row in manifest.iterrows()}

    def loader(subject: str, candidate: CrossPersonCandidate) -> SubjectFeatureMatrix:
        row = rows[str(subject)]
        current_label_column = label_column or str(row.get("label_column", ""))
        if not current_label_column:
            raise ValueError("Provide --label-column or a label_column column in the manifest.")
        metadata_value = row.get(metadata_csv_column, "") if metadata_csv_column in manifest.columns else ""
        calibration_epochs_value = row.get(calibration_epochs_column, "") if calibration_epochs_column in manifest.columns else ""
        calibration_metadata_value = row.get(calibration_metadata_csv_column, "") if calibration_metadata_csv_column in manifest.columns else ""
        return load_mne_subject_features(
            Path(row[epochs_column]),
            candidate=candidate,
            subject=str(subject),
            label_column=current_label_column,
            metadata_csv=_optional_path(metadata_value),
            picks=picks,
            tmin=tmin,
            tmax=tmax,
            calibration_epochs_path=_optional_path(calibration_epochs_value),
            calibration_metadata_csv=_optional_path(calibration_metadata_value),
        )

    return subjects, loader


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    return None if not text else Path(text)


def export_nested_cross_person(
    artifacts: NestedCrossPersonArtifacts,
    *,
    out_dir: Path,
    prefix: str = "cross_person",
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "outer": out_dir / f"{prefix}_outer.csv",
        "inner_validation": out_dir / f"{prefix}_inner_validation.csv",
        "selected": out_dir / f"{prefix}_selected.csv",
        "predictions": out_dir / f"{prefix}_predictions.csv",
    }
    for key, path in paths.items():
        rows = artifacts.as_dict()[key]
        pd.DataFrame(rows).to_csv(path, index=False)
    return paths


def _parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(token.strip()) for token in value.split(",") if token.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one numeric value is required.")
    return values


def _parse_token_list(value: str) -> tuple[str, ...]:
    values = tuple(token.strip() for token in value.split(",") if token.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one value is required.")
    return values


def _parse_pca_components_list(value: str) -> tuple[int | float | str | None, ...]:
    values: list[int | float | str | None] = []
    for token in value.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if lowered in {"none", "auto", "default"}:
            values.append(None)
        else:
            try:
                parsed: int | float = float(stripped) if any(marker in stripped for marker in (".", "e", "E")) else int(stripped)
            except ValueError:
                values.append(stripped)
            else:
                values.append(parsed)
    if not values:
        raise argparse.ArgumentTypeError("At least one PCA component value is required.")
    return tuple(values)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nested cross-person decoding from an MNE Epochs manifest.")
    parser.add_argument("--manifest", type=Path, required=True, help="CSV with subject, epochs, optional metadata_csv, and optional calibration columns.")
    parser.add_argument("--label-column", help="Metadata label column. Overrides per-row label_column values when set.")
    parser.add_argument("--subject-column", default="subject")
    parser.add_argument("--epochs-column", default="epochs")
    parser.add_argument("--metadata-csv-column", default="metadata_csv")
    parser.add_argument("--calibration-epochs-column", default="calibration_epochs")
    parser.add_argument("--calibration-metadata-csv-column", default="calibration_metadata_csv")
    parser.add_argument("--picks", default="data")
    parser.add_argument("--tmin", type=float)
    parser.add_argument("--tmax", type=float)
    parser.add_argument("--outer-subjects", type=_parse_token_list, help="Comma-separated subset of held-out subjects to evaluate.")
    parser.add_argument("--window-centers", type=_parse_float_list, default=DEFAULT_WINDOW_CENTERS)
    parser.add_argument("--window-size", type=float, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--baseline-window", type=float, nargs=2, default=DEFAULT_BASELINE_WINDOW)
    parser.add_argument("--feature-modes", type=_parse_token_list, default=("sensor_flat",))
    parser.add_argument("--normalizations", type=_parse_token_list, default=("subject_baseline_z",))
    parser.add_argument("--alignments", type=_parse_token_list, default=("none",))
    parser.add_argument("--decoders", type=_parse_token_list, default=("linear_svm",))
    parser.add_argument("--emission-modes", type=_parse_token_list, default=("calibrated",))
    parser.add_argument("--feature-preprocessors", type=_parse_token_list, default=("pca_whiten",))
    parser.add_argument("--pca-components-values", type=_parse_pca_components_list, default=(64,))
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--max-trials-per-class-per-subject", type=int)
    parser.add_argument("--selection-metric", choices=("accuracy", "balanced_accuracy", "log_loss"), default="balanced_accuracy")
    parser.add_argument("--selection-ensemble-size", type=int, default=1)
    parser.add_argument("--selection-ensemble-diversity", choices=SELECTION_DIVERSITY_MODES, default="none")
    parser.add_argument("--label-shuffle-control", action="store_true")
    parser.add_argument("--label-shuffle-seed", type=int, default=0)
    parser.add_argument("--target-calibration-label-shuffle-control", action="store_true")
    parser.add_argument("--target-calibration-label-shuffle-seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="cross_person")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    manifest = pd.read_csv(args.manifest)
    subjects, loader = make_manifest_feature_loader(
        manifest,
        label_column=args.label_column,
        subject_column=args.subject_column,
        epochs_column=args.epochs_column,
        metadata_csv_column=args.metadata_csv_column,
        calibration_epochs_column=args.calibration_epochs_column,
        calibration_metadata_csv_column=args.calibration_metadata_csv_column,
        picks=args.picks,
        tmin=args.tmin,
        tmax=args.tmax,
    )
    candidates = make_cross_person_candidate_grid(
        window_centers=args.window_centers,
        window_size=args.window_size,
        baseline_window=tuple(args.baseline_window),
        feature_modes=args.feature_modes,
        normalizations=args.normalizations,
        alignments=args.alignments,
        decoders=args.decoders,
        emission_modes=args.emission_modes,
        feature_preprocessors=args.feature_preprocessors,
        pca_components_values=args.pca_components_values,
        max_iter=args.max_iter,
        max_trials_per_class_per_subject=args.max_trials_per_class_per_subject,
    )
    artifacts = run_nested_cross_person_from_loader(
        subjects,
        candidate_configs=candidates,
        feature_loader=loader,
        outer_subjects=args.outer_subjects,
        selection_metric=args.selection_metric,
        selection_ensemble_size=args.selection_ensemble_size,
        selection_ensemble_diversity=args.selection_ensemble_diversity,
        label_shuffle_control=args.label_shuffle_control,
        label_shuffle_seed=args.label_shuffle_seed,
        target_calibration_label_shuffle_control=args.target_calibration_label_shuffle_control,
        target_calibration_label_shuffle_seed=args.target_calibration_label_shuffle_seed,
        progress=lambda message: print(message, flush=True),
    )
    paths = export_nested_cross_person(artifacts, out_dir=args.out_dir, prefix=args.prefix)
    for name, path in paths.items():
        print(f"Wrote {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
