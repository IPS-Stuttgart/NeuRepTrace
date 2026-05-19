"""Cross-subject decoding helpers for precomputed participant feature matrices.

This module keeps dataset-specific loading, feature extraction, and paper exports
outside NeuRepTrace while making the leakage-sensitive evaluation loop reusable:
feature providers pass one ``ParticipantFeatureSet`` per subject, then NeuRepTrace
owns train/test partitioning, model fitting, nested LOSO selection, score tables,
and prediction-row bookkeeping.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from neureptrace.decoding.class_scores import predict_window_class_scores
from neureptrace.decoding.windowed import FitModel, WindowedModelBundle, score_windowed_decoding

SELECTION_METRICS = ("balanced_accuracy", "accuracy")


@dataclass(frozen=True)
class ParticipantFeatureSet:
    """Decoding-ready feature matrix for one participant or subject.

    ``features`` must be a two-dimensional array with one row per trial or
    observation. ``labels`` must contain one label per feature row. Optional
    ``sample_ids`` are propagated into prediction rows, which lets downstream
    dataset packages keep links to original trial numbers without making them
    part of the generic evaluation algorithm.
    """

    participant: Hashable
    features: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecoderCandidate:
    """One model candidate for nested cross-subject selection."""

    name: str
    fit_model: FitModel
    components_pca: int | float = float("inf")
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CrossSubjectFoldResult:
    """One held-out-participant score from a cross-subject evaluation."""

    fold_index: int
    test_participant: Hashable
    train_participants: tuple[Hashable, ...]
    true_labels: np.ndarray
    predictions: np.ndarray
    scores: np.ndarray
    accuracy: float
    balanced_accuracy: float
    chance_accuracy: float
    permutation_accuracy: np.ndarray
    permutation_p_value: float
    model_bundle: WindowedModelBundle
    sample_ids: np.ndarray
    class_scores: np.ndarray | None = None
    score_classes: np.ndarray | None = None
    true_label_ranks: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def mean_true_label_rank(self) -> float:
        """Mean rank of the true class among scoreable rows."""

        finite = self.true_label_ranks[np.isfinite(self.true_label_ranks)]
        if finite.size == 0:
            return np.nan
        return float(np.mean(finite))

    def topk_accuracy(self, k: int) -> float:
        """Return the fraction of rows whose true label is ranked in the top ``k``."""

        if k < 1:
            raise ValueError("k must be at least 1.")
        if self.true_label_ranks.size == 0:
            return np.nan
        return float(np.mean(self.true_label_ranks <= k))


@dataclass(frozen=True)
class CrossSubjectEvaluationResult:
    """Collection of LOSO folds plus convenience row exporters."""

    folds: tuple[CrossSubjectFoldResult, ...]
    label_shuffle_control: bool = False
    label_shuffle_seed: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def mean_accuracy(self) -> float:
        return _nanmean([fold.accuracy for fold in self.folds])

    @property
    def mean_balanced_accuracy(self) -> float:
        return _nanmean([fold.balanced_accuracy for fold in self.folds])

    def fold_rows(self) -> list[dict[str, Any]]:
        """Return compact per-held-out-participant score rows."""

        rows: list[dict[str, Any]] = []
        for fold in self.folds:
            rows.append(
                {
                    **dict(fold.metadata),
                    "fold_index": fold.fold_index,
                    "test_participant": fold.test_participant,
                    "train_participants": ";".join(str(participant) for participant in fold.train_participants),
                    "n_train_participants": len(fold.train_participants),
                    "n_test_samples": int(fold.true_labels.shape[0]),
                    "accuracy": fold.accuracy,
                    "balanced_accuracy": fold.balanced_accuracy,
                    "chance_accuracy": fold.chance_accuracy,
                    "top2_accuracy": fold.topk_accuracy(2),
                    "top3_accuracy": fold.topk_accuracy(3),
                    "mean_true_label_rank": fold.mean_true_label_rank,
                    "permutation_p_value": fold.permutation_p_value,
                    "label_shuffle_control": self.label_shuffle_control,
                    "label_shuffle_seed": "" if self.label_shuffle_seed is None else int(self.label_shuffle_seed),
                }
            )
        return rows

    def prediction_rows(self) -> list[dict[str, Any]]:
        """Return one row per held-out prediction."""

        rows: list[dict[str, Any]] = []
        for fold in self.folds:
            for row_index, (sample_id, true_label, predicted_label, score) in enumerate(
                zip(fold.sample_ids, fold.true_labels, fold.predictions, fold.scores, strict=True)
            ):
                rank = fold.true_label_ranks[row_index] if row_index < fold.true_label_ranks.shape[0] else np.nan
                rows.append(
                    {
                        **dict(fold.metadata),
                        "fold_index": fold.fold_index,
                        "test_participant": fold.test_participant,
                        "sample_id": _python_scalar(sample_id),
                        "test_row_index": row_index,
                        "true_label": _python_scalar(true_label),
                        "predicted_label": _python_scalar(predicted_label),
                        "correct": bool(predicted_label == true_label),
                        "score": float(score) if np.isfinite(score) else np.nan,
                        "true_label_rank": float(rank) if np.isfinite(rank) else np.nan,
                        "top2_correct": bool(np.isfinite(rank) and rank <= 2),
                        "top3_correct": bool(np.isfinite(rank) and rank <= 3),
                        "label_shuffle_control": self.label_shuffle_control,
                        "label_shuffle_seed": "" if self.label_shuffle_seed is None else int(self.label_shuffle_seed),
                    }
                )
        return rows


@dataclass(frozen=True)
class CandidateScore:
    """Inner-LOSO score for one candidate inside one outer fold."""

    outer_test_participant: Hashable
    candidate_index: int
    candidate_name: str
    evaluation: CrossSubjectEvaluationResult = field(repr=False)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def mean_accuracy(self) -> float:
        return self.evaluation.mean_accuracy

    @property
    def mean_balanced_accuracy(self) -> float:
        return self.evaluation.mean_balanced_accuracy


@dataclass(frozen=True)
class NestedOuterFoldResult:
    """Selected candidate and untouched outer-fold score."""

    outer_fold_index: int
    selected_candidate_index: int
    selected_candidate_name: str
    selected_metric_value: float
    fold: CrossSubjectFoldResult
    inner_candidate_scores: tuple[CandidateScore, ...]


@dataclass(frozen=True)
class NestedCrossSubjectResult:
    """Nested LOSO model-selection result."""

    outer_folds: tuple[NestedOuterFoldResult, ...]
    selection_metric: str = "balanced_accuracy"
    label_shuffle_control: bool = False
    label_shuffle_seed: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def folds(self) -> tuple[CrossSubjectFoldResult, ...]:
        return tuple(outer.fold for outer in self.outer_folds)

    @property
    def mean_accuracy(self) -> float:
        return _nanmean([fold.accuracy for fold in self.folds])

    @property
    def mean_balanced_accuracy(self) -> float:
        return _nanmean([fold.balanced_accuracy for fold in self.folds])

    def fold_rows(self) -> list[dict[str, Any]]:
        """Return compact per-outer-fold score rows with selected-candidate metadata."""

        rows: list[dict[str, Any]] = []
        for outer in self.outer_folds:
            fold = outer.fold
            rows.append(
                {
                    **dict(fold.metadata),
                    "outer_fold_index": outer.outer_fold_index,
                    "test_participant": fold.test_participant,
                    "selected_candidate_index": outer.selected_candidate_index,
                    "selected_candidate_name": outer.selected_candidate_name,
                    "selection_metric": self.selection_metric,
                    "selected_metric_value": outer.selected_metric_value,
                    "accuracy": fold.accuracy,
                    "balanced_accuracy": fold.balanced_accuracy,
                    "chance_accuracy": fold.chance_accuracy,
                    "top2_accuracy": fold.topk_accuracy(2),
                    "top3_accuracy": fold.topk_accuracy(3),
                    "mean_true_label_rank": fold.mean_true_label_rank,
                    "label_shuffle_control": self.label_shuffle_control,
                    "label_shuffle_seed": "" if self.label_shuffle_seed is None else int(self.label_shuffle_seed),
                }
            )
        return rows

    def selection_rows(self) -> list[dict[str, Any]]:
        """Return one row per candidate considered inside each outer fold."""

        rows: list[dict[str, Any]] = []
        for outer in self.outer_folds:
            for candidate_score in outer.inner_candidate_scores:
                rows.append(
                    {
                        **dict(candidate_score.metadata),
                        "outer_fold_index": outer.outer_fold_index,
                        "outer_test_participant": candidate_score.outer_test_participant,
                        "candidate_index": candidate_score.candidate_index,
                        "candidate_name": candidate_score.candidate_name,
                        "selection_metric": self.selection_metric,
                        "mean_accuracy": candidate_score.mean_accuracy,
                        "mean_balanced_accuracy": candidate_score.mean_balanced_accuracy,
                        "selected": candidate_score.candidate_index == outer.selected_candidate_index,
                    }
                )
        return rows

    def prediction_rows(self) -> list[dict[str, Any]]:
        """Return one row per prediction from the selected outer-fold models."""

        evaluation = CrossSubjectEvaluationResult(
            self.folds,
            label_shuffle_control=self.label_shuffle_control,
            label_shuffle_seed=self.label_shuffle_seed,
            metadata=self.metadata,
        )
        return evaluation.prediction_rows()


def leave_one_subject_out(
    feature_sets: Sequence[ParticipantFeatureSet],
    *,
    fit_model: FitModel,
    components_pca: int | float = float("inf"),
    train_window: tuple[float, float] | None = None,
    chance_accuracy: float | None = None,
    include_class_scores: bool = True,
    predict_fallback_for_class_scores: bool = True,
    label_shuffle_control: bool = False,
    label_shuffle_seed: int | None = 0,
    n_permutations: int = 0,
    permutation_seed: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CrossSubjectEvaluationResult:
    """Evaluate one model by holding out each participant exactly once.

    The caller supplies decoding-ready features. NeuRepTrace concatenates all
    non-held-out participants for training, fits a fresh fold-local model, and
    scores the untouched held-out participant. PCA and permutations are delegated
    to the existing windowed-decoding helpers, so transforms remain fold-local.
    """

    normalized_sets = _validate_feature_sets(feature_sets, min_sets=2)
    folds = []
    for fold_index, test_set in enumerate(normalized_sets):
        train_sets = tuple(feature_set for index, feature_set in enumerate(normalized_sets) if index != fold_index)
        folds.append(
            _score_train_test_fold(
                train_sets,
                test_set,
                fold_index=fold_index,
                fit_model=fit_model,
                components_pca=components_pca,
                train_window=train_window,
                chance_accuracy=chance_accuracy,
                include_class_scores=include_class_scores,
                predict_fallback_for_class_scores=predict_fallback_for_class_scores,
                label_shuffle_control=label_shuffle_control,
                label_shuffle_seed=label_shuffle_seed,
                n_permutations=n_permutations,
                permutation_seed=permutation_seed,
                metadata=metadata,
            )
        )
    return CrossSubjectEvaluationResult(
        tuple(folds),
        label_shuffle_control=label_shuffle_control,
        label_shuffle_seed=label_shuffle_seed,
        metadata={} if metadata is None else dict(metadata),
    )


def nested_leave_one_subject_out(
    feature_sets: Sequence[ParticipantFeatureSet],
    *,
    candidates: Sequence[DecoderCandidate],
    selection_metric: str = "balanced_accuracy",
    outer_participants: Sequence[Hashable] | None = None,
    train_window: tuple[float, float] | None = None,
    chance_accuracy: float | None = None,
    include_class_scores: bool = True,
    predict_fallback_for_class_scores: bool = True,
    label_shuffle_control: bool = False,
    label_shuffle_seed: int | None = 0,
    n_permutations: int = 0,
    permutation_seed: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> NestedCrossSubjectResult:
    """Run nested LOSO selection and score each outer participant once.

    For every outer participant, candidates are compared by an inner LOSO over
    the remaining participants only. The selected candidate is then fit on all
    non-outer participants and evaluated on the untouched outer participant.
    Ties are resolved by candidate order, making selection deterministic.
    """

    normalized_sets = _validate_feature_sets(feature_sets, min_sets=3)
    normalized_candidates = _validate_candidates(candidates)
    selection_metric = _normalize_selection_metric(selection_metric)
    outer_indices = _outer_indices(normalized_sets, outer_participants)
    outer_results: list[NestedOuterFoldResult] = []

    for outer_fold_index, outer_index in enumerate(outer_indices):
        outer_test_set = normalized_sets[outer_index]
        inner_sets = tuple(feature_set for index, feature_set in enumerate(normalized_sets) if index != outer_index)
        candidate_scores = []
        for candidate_index, candidate in enumerate(normalized_candidates):
            candidate_evaluation = leave_one_subject_out(
                inner_sets,
                fit_model=candidate.fit_model,
                components_pca=candidate.components_pca,
                train_window=train_window,
                chance_accuracy=chance_accuracy,
                include_class_scores=include_class_scores,
                predict_fallback_for_class_scores=predict_fallback_for_class_scores,
                label_shuffle_control=label_shuffle_control,
                label_shuffle_seed=label_shuffle_seed,
                n_permutations=0,
                metadata={"candidate_index": candidate_index, "candidate_name": candidate.name, **dict(candidate.metadata)},
            )
            candidate_scores.append(
                CandidateScore(
                    outer_test_participant=outer_test_set.participant,
                    candidate_index=candidate_index,
                    candidate_name=candidate.name,
                    evaluation=candidate_evaluation,
                    metadata=candidate.metadata,
                )
            )

        selected = _select_candidate(candidate_scores, selection_metric)
        selected_candidate = normalized_candidates[selected.candidate_index]
        outer_fold = _score_train_test_fold(
            inner_sets,
            outer_test_set,
            fold_index=outer_fold_index,
            fit_model=selected_candidate.fit_model,
            components_pca=selected_candidate.components_pca,
            train_window=train_window,
            chance_accuracy=chance_accuracy,
            include_class_scores=include_class_scores,
            predict_fallback_for_class_scores=predict_fallback_for_class_scores,
            label_shuffle_control=label_shuffle_control,
            label_shuffle_seed=label_shuffle_seed,
            n_permutations=n_permutations,
            permutation_seed=permutation_seed,
            metadata={
                "selected_candidate_index": selected.candidate_index,
                "selected_candidate_name": selected.candidate_name,
                **dict(selected_candidate.metadata),
            },
        )
        outer_results.append(
            NestedOuterFoldResult(
                outer_fold_index=outer_fold_index,
                selected_candidate_index=selected.candidate_index,
                selected_candidate_name=selected.candidate_name,
                selected_metric_value=_candidate_metric(selected, selection_metric),
                fold=outer_fold,
                inner_candidate_scores=tuple(candidate_scores),
            )
        )

    return NestedCrossSubjectResult(
        tuple(outer_results),
        selection_metric=selection_metric,
        label_shuffle_control=label_shuffle_control,
        label_shuffle_seed=label_shuffle_seed,
        metadata={} if metadata is None else dict(metadata),
    )


def summarize_cross_subject_folds(result: CrossSubjectEvaluationResult | NestedCrossSubjectResult | Sequence[CrossSubjectFoldResult]) -> dict[str, Any]:
    """Summarize held-out participant folds with subject-level descriptive stats."""

    folds = _as_fold_tuple(result)
    if not folds:
        return {}
    accuracy = np.asarray([fold.accuracy for fold in folds], dtype=float)
    balanced = np.asarray([fold.balanced_accuracy for fold in folds], dtype=float)
    chance = np.asarray([fold.chance_accuracy for fold in folds], dtype=float)
    top2 = np.asarray([fold.topk_accuracy(2) for fold in folds], dtype=float)
    top3 = np.asarray([fold.topk_accuracy(3) for fold in folds], dtype=float)
    mean_ranks = np.asarray([fold.mean_true_label_rank for fold in folds], dtype=float)
    differences = balanced - chance
    return {
        "n_outer_folds": len(folds),
        "n_test_participants": len({fold.test_participant for fold in folds}),
        "accuracy_mean": float(np.mean(accuracy)),
        "accuracy_median": float(np.median(accuracy)),
        "accuracy_sem": _sem(accuracy),
        "balanced_accuracy_mean": float(np.mean(balanced)),
        "balanced_accuracy_median": float(np.median(balanced)),
        "balanced_accuracy_sem": _sem(balanced),
        "chance_accuracy_mean": float(np.mean(chance)),
        "top2_accuracy_mean": _nanmean(top2),
        "top3_accuracy_mean": _nanmean(top3),
        "mean_true_label_rank_mean": _nanmean(mean_ranks),
        "mean_above_chance": float(np.mean(differences)),
        "participants_above_chance": int(np.sum(differences > 0.0)),
        "participants_at_or_below_chance": int(np.sum(differences <= 0.0)),
        "participants_total": int(differences.shape[0]),
    }


def stack_feature_sets(feature_sets: Sequence[ParticipantFeatureSet]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate feature sets and return features, labels, and participant ids."""

    normalized_sets = _validate_feature_sets(feature_sets, min_sets=1)
    features = np.vstack([feature_set.features for feature_set in normalized_sets])
    labels = np.concatenate([feature_set.labels for feature_set in normalized_sets])
    participants = np.concatenate(
        [np.asarray([feature_set.participant] * feature_set.labels.shape[0], dtype=object) for feature_set in normalized_sets]
    )
    return features, labels, participants


def chance_accuracy_from_labels(labels: Sequence | np.ndarray) -> float:
    """Return uniform-class chance accuracy implied by a label vector."""

    labels = np.asarray(labels).ravel()
    classes = np.unique(labels)
    if classes.size == 0:
        raise ValueError("At least one label is required to compute chance accuracy.")
    return float(1.0 / classes.size)


def rank_true_labels(true_labels: Sequence | np.ndarray, class_scores: Sequence[Sequence[float]] | np.ndarray, score_classes: Sequence | np.ndarray) -> np.ndarray:
    """Return one-based ranks of true labels under per-class score rows.

    A rank of 1 means no class has a strictly higher score than the true class.
    Missing true classes or non-finite true-class scores yield ``NaN``.
    """

    true_labels = np.asarray(true_labels).ravel()
    class_scores = np.asarray(class_scores, dtype=float)
    score_classes = np.asarray(score_classes).ravel()
    if class_scores.ndim != 2:
        raise ValueError("class_scores must be a two-dimensional matrix.")
    if class_scores.shape[0] != true_labels.shape[0]:
        raise ValueError("class_scores rows must match true_labels length.")
    if class_scores.shape[1] != score_classes.shape[0]:
        raise ValueError("class_scores columns must match score_classes length.")

    ranks = np.full(true_labels.shape[0], np.nan, dtype=float)
    for row_index, (label, row_scores) in enumerate(zip(true_labels, class_scores, strict=True)):
        matches = np.flatnonzero(score_classes == label)
        if matches.size == 0:
            continue
        true_score = row_scores[int(matches[0])]
        if not np.isfinite(true_score):
            continue
        ranks[row_index] = float(1 + np.sum(row_scores > true_score))
    return ranks


def _score_train_test_fold(
    train_sets: Sequence[ParticipantFeatureSet],
    test_set: ParticipantFeatureSet,
    *,
    fold_index: int,
    fit_model: FitModel,
    components_pca: int | float,
    train_window: tuple[float, float] | None,
    chance_accuracy: float | None,
    include_class_scores: bool,
    predict_fallback_for_class_scores: bool,
    label_shuffle_control: bool,
    label_shuffle_seed: int | None,
    n_permutations: int,
    permutation_seed: int | None,
    metadata: Mapping[str, Any] | None,
) -> CrossSubjectFoldResult:
    normalized_train_sets = _validate_feature_sets(train_sets, min_sets=1)
    train_features, train_labels, _train_participant_rows = stack_feature_sets(normalized_train_sets)
    test_set = _normalize_feature_set(test_set)
    fit_labels = _shuffle_labels(train_labels, label_shuffle_seed, fold_index) if label_shuffle_control else train_labels
    permutation_rng = _rng_from_seed(permutation_seed, fold_index) if n_permutations > 0 else None
    windowed_result = score_windowed_decoding(
        train_features,
        fit_labels,
        test_set.features,
        test_set.labels,
        fit_model=fit_model,
        components_pca=components_pca,
        train_window=train_window,
        n_permutations=n_permutations,
        permutation_rng=permutation_rng,
    )
    class_scores = None
    score_classes = None
    true_label_ranks = np.full(test_set.labels.shape[0], np.nan, dtype=float)
    if include_class_scores:
        class_scores, score_classes = predict_window_class_scores(
            windowed_result.model_bundle,
            test_set.features,
            predict_fallback=predict_fallback_for_class_scores,
        )
        if class_scores is not None and score_classes is not None:
            true_label_ranks = rank_true_labels(test_set.labels, class_scores, score_classes)

    return CrossSubjectFoldResult(
        fold_index=fold_index,
        test_participant=test_set.participant,
        train_participants=tuple(feature_set.participant for feature_set in normalized_train_sets),
        true_labels=test_set.labels,
        predictions=windowed_result.predictions,
        scores=windowed_result.scores,
        accuracy=float(accuracy_score(test_set.labels, windowed_result.predictions)),
        balanced_accuracy=_balanced_accuracy(test_set.labels, windowed_result.predictions),
        chance_accuracy=float(chance_accuracy) if chance_accuracy is not None else chance_accuracy_from_labels(train_labels),
        permutation_accuracy=windowed_result.permutation_accuracy,
        permutation_p_value=windowed_result.permutation_p_value,
        model_bundle=windowed_result.model_bundle,
        sample_ids=_sample_ids(test_set),
        class_scores=class_scores,
        score_classes=score_classes,
        true_label_ranks=true_label_ranks,
        metadata={} if metadata is None else dict(metadata),
    )


def _validate_feature_sets(feature_sets: Sequence[ParticipantFeatureSet], *, min_sets: int) -> tuple[ParticipantFeatureSet, ...]:
    normalized_sets = tuple(_normalize_feature_set(feature_set) for feature_set in feature_sets)
    if len(normalized_sets) < min_sets:
        raise ValueError(f"At least {min_sets} participant feature sets are required.")
    participants = [feature_set.participant for feature_set in normalized_sets]
    if len(set(participants)) != len(participants):
        raise ValueError("Participant identifiers must be unique within one cross-subject evaluation.")
    return normalized_sets


def _normalize_feature_set(feature_set: ParticipantFeatureSet) -> ParticipantFeatureSet:
    features = np.asarray(feature_set.features, dtype=float)
    if features.ndim != 2:
        raise ValueError("ParticipantFeatureSet.features must be a two-dimensional matrix.")
    if features.shape[0] == 0:
        raise ValueError("ParticipantFeatureSet.features must contain at least one row.")
    labels = np.asarray(feature_set.labels).ravel()
    if labels.shape[0] != features.shape[0]:
        raise ValueError("ParticipantFeatureSet.labels must contain one label per feature row.")
    sample_ids = None if feature_set.sample_ids is None else np.asarray(feature_set.sample_ids).ravel()
    if sample_ids is not None and sample_ids.shape[0] != features.shape[0]:
        raise ValueError("ParticipantFeatureSet.sample_ids must contain one id per feature row.")
    return ParticipantFeatureSet(
        participant=_python_scalar(feature_set.participant),
        features=features,
        labels=labels,
        sample_ids=sample_ids,
        metadata={} if feature_set.metadata is None else dict(feature_set.metadata),
    )


def _validate_candidates(candidates: Sequence[DecoderCandidate]) -> tuple[DecoderCandidate, ...]:
    normalized = tuple(candidates)
    if not normalized:
        raise ValueError("At least one DecoderCandidate is required for nested LOSO selection.")
    for index, candidate in enumerate(normalized):
        if not candidate.name:
            raise ValueError(f"Candidate {index} has an empty name.")
        if not callable(candidate.fit_model):
            raise ValueError(f"Candidate {candidate.name!r} fit_model is not callable.")
    return normalized


def _outer_indices(feature_sets: Sequence[ParticipantFeatureSet], outer_participants: Sequence[Hashable] | None) -> tuple[int, ...]:
    if outer_participants is None:
        return tuple(range(len(feature_sets)))
    participant_to_index = {feature_set.participant: index for index, feature_set in enumerate(feature_sets)}
    indices = []
    for participant in outer_participants:
        participant = _python_scalar(participant)
        if participant not in participant_to_index:
            raise ValueError(f"Unknown outer participant: {participant!r}")
        indices.append(participant_to_index[participant])
    return tuple(indices)


def _normalize_selection_metric(metric: str) -> str:
    normalized = metric.lower().replace("-", "_")
    if normalized not in SELECTION_METRICS:
        raise ValueError(f"selection_metric must be one of {SELECTION_METRICS}.")
    return normalized


def _select_candidate(candidate_scores: Sequence[CandidateScore], selection_metric: str) -> CandidateScore:
    return max(candidate_scores, key=lambda score: (_finite_or_negative_infinity(_candidate_metric(score, selection_metric)), -score.candidate_index))


def _candidate_metric(candidate_score: CandidateScore, selection_metric: str) -> float:
    if selection_metric == "balanced_accuracy":
        return candidate_score.mean_balanced_accuracy
    if selection_metric == "accuracy":
        return candidate_score.mean_accuracy
    raise ValueError(f"Unsupported selection metric: {selection_metric}")


def _shuffle_labels(labels: np.ndarray, seed: int | None, fold_index: int) -> np.ndarray:
    shuffled = np.array(labels, copy=True)
    _rng_from_seed(seed, fold_index).shuffle(shuffled)
    return shuffled


def _rng_from_seed(seed: int | None, *salts: int) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    seed_values = [int(seed), *(int(salt) for salt in salts)]
    return np.random.default_rng(np.random.SeedSequence(seed_values))


def _sample_ids(feature_set: ParticipantFeatureSet) -> np.ndarray:
    if feature_set.sample_ids is None:
        return np.arange(feature_set.labels.shape[0], dtype=int)
    return np.asarray(feature_set.sample_ids).ravel()


def _balanced_accuracy(true_labels: np.ndarray, predictions: np.ndarray) -> float:
    if np.unique(true_labels).shape[0] < 2:
        return float(accuracy_score(true_labels, predictions))
    return float(balanced_accuracy_score(true_labels, predictions))


def _as_fold_tuple(result: CrossSubjectEvaluationResult | NestedCrossSubjectResult | Sequence[CrossSubjectFoldResult]) -> tuple[CrossSubjectFoldResult, ...]:
    if isinstance(result, CrossSubjectEvaluationResult):
        return result.folds
    if isinstance(result, NestedCrossSubjectResult):
        return result.folds
    return tuple(result)


def _sem(values: Sequence[float] | np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return np.nan
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def _nanmean(values: Sequence[float] | np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.all(~np.isfinite(values)):
        return np.nan
    return float(np.nanmean(values))


def _finite_or_negative_infinity(value: float) -> float:
    return float(value) if np.isfinite(value) else float("-inf")


def _python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value
