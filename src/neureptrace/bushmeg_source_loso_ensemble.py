"""Strict source-only nested top-k ensemble decoding for BUSH-MEG.

The runner reuses ``bushmeg_source_loso`` data loading and candidate features,
keeps cue files out of the workflow, scores candidates by inner source-subject
LOSO only, and refits a weighted probability ensemble on all source subjects for
the held-out participant.  It is intended to reduce brittle single-candidate
selection among near-tied BUSH-MEG source-only decoders.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from neureptrace.bushmeg_source_loso import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_SELECTION_METRIC,
    MINIMIZE_SELECTION_METRICS,
    SUPPORTED_SELECTION_METRICS,
    CandidateSpec,
    FeatureCache,
    SubjectEpochs,
    _apply_class_bias,
    _candidate_grid,
    _candidate_metrics,
    _candidate_rowspec,
    _fit_class_bias,
    _inner_loso_scores,
    _load_subjects_from_config,
    _predict_candidate,
    _resolve_output,
    _section,
)
from neureptrace.bushmeg_cue_source_weighting import (
    DEFAULT_CUE_SOURCE_TEMPERATURE,
    DEFAULT_CUE_SOURCE_WEIGHTING,
    cue_source_weights_from_summaries,
    load_cue_summaries_from_config,
    normalize_cue_source_temperature,
    normalize_cue_source_weighting,
)
from neureptrace.dataset_config import apply_overrides, load_config
from neureptrace.bushmeg_cue_source_weights import (
    CueSourceWeights,
    resolve_cue_source_weights,
    write_cue_source_weight_csv,
)

DEFAULT_ENSEMBLE_TOP_K = 5
DEFAULT_ENSEMBLE_WEIGHTING = "softmax"
DEFAULT_ENSEMBLE_TEMPERATURE = 0.01
DEFAULT_ENSEMBLE_CLASS_BIAS = "none"
DEFAULT_STACKING_MAX_ITER = 250
DEFAULT_STACKING_LEARNING_RATE = 0.25
DEFAULT_STACKING_EPSILON = 1.0e-12
DEFAULT_RERANK_TOP_K = 0
DEFAULT_RERANK_ALPHA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)
ENSEMBLE_WEIGHTING_MODES = {"uniform", "rank", "softmax", "stacked"}
ENSEMBLE_CLASS_BIAS_MODES = {"none", "log_prior", "balanced_accuracy"}


@dataclass(frozen=True, slots=True)
class CandidateSummary:
    candidate: CandidateSpec
    mean_score: float
    std_score: float
    comparable_score: float
    n_folds: int


@dataclass(frozen=True, slots=True)
class EnsembleMember:
    candidate: CandidateSpec
    rank: int
    weight: float
    mean_score: float
    std_score: float
    comparable_score: float


@dataclass(frozen=True, slots=True)
class TopKPairwiseReranker:
    n_classes: int
    top_k: int
    alpha: float
    intercepts: tuple[float, ...]
    slopes: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class EnsembleSelection:
    members: tuple[EnsembleMember, ...]
    selection_metric: str
    weighting: str
    temperature: float | None
    class_bias_mode: str = "none"
    class_bias: tuple[float, ...] = ()
    reranker: TopKPairwiseReranker | None = None
    oof_balanced_accuracy: float | None = None
    oof_log_loss: float | None = None

    @property
    def name(self) -> str:
        bias_suffix = "" if self.class_bias_mode == "none" else f"__bias_{self.class_bias_mode}"
        rerank_suffix = "" if self.reranker is None else f"__rerank_top{self.reranker.top_k}_a{self.reranker.alpha:g}"
        return f"ensemble_top{len(self.members)}__{self.weighting}{bias_suffix}{rerank_suffix}"


def _larger_is_better(score: float, metric: str) -> float:
    return -float(score) if metric in MINIMIZE_SELECTION_METRICS else float(score)


def _normalize_weighting(value: Any) -> str:
    mode = DEFAULT_ENSEMBLE_WEIGHTING if value is None else str(value).strip().lower().replace("-", "_")
    mode = {"average": "uniform", "mean": "uniform", "equal": "uniform", "inverse_rank": "rank", "rank_weighted": "rank", "exp": "softmax"}.get(mode, mode)
    if mode not in ENSEMBLE_WEIGHTING_MODES:
        raise ValueError(f"Unknown ensemble weighting {value!r}; choose one of {sorted(ENSEMBLE_WEIGHTING_MODES)}.")
    return mode


def _normalize_temperature(value: Any, weighting: str) -> float | None:
    if weighting != "softmax":
        return None
    temperature = DEFAULT_ENSEMBLE_TEMPERATURE if value is None else float(value)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("source_loso.ensemble_temperature must be positive and finite.")
    return temperature


def _normalize_ensemble_class_bias(value: Any) -> str:
    mode = DEFAULT_ENSEMBLE_CLASS_BIAS if value is None else str(value).strip().lower().replace("-", "_")
    mode = {
        "": "none",
        "false": "none",
        "off": "none",
        "no": "none",
        "prior": "log_prior",
        "class_prior": "log_prior",
        "balanced_acc": "balanced_accuracy",
        "source_balanced_accuracy": "balanced_accuracy",
        "source_loso_balanced_accuracy": "balanced_accuracy",
    }.get(mode, mode)
    if mode not in ENSEMBLE_CLASS_BIAS_MODES:
        raise ValueError(f"Unknown ensemble class-bias mode {value!r}; choose one of {sorted(ENSEMBLE_CLASS_BIAS_MODES)}.")
    return mode


def _normalize_rerank_top_k(value: Any) -> int:
    if value is None:
        return DEFAULT_RERANK_TOP_K
    if isinstance(value, str) and value.strip().lower().replace("-", "_") in {"", "none", "off", "false", "no"}:
        return 0
    top_k = int(value)
    if top_k < 0:
        raise ValueError("source_loso.rerank_top_k must be non-negative; use 0 to disable reranking.")
    return top_k


def _parse_float_grid(value: Any, default: Sequence[float]) -> list[float]:
    if value is None:
        values = [float(item) for item in default]
    elif isinstance(value, str):
        tokens = [token.strip() for token in value.split(",") if token.strip()]
        values = [float(token) for token in tokens]
    elif isinstance(value, Sequence):
        values = [float(item) for item in value]
    else:
        values = [float(value)]
    if not values or not np.all(np.isfinite(values)):
        raise ValueError("Reranker alpha grid must contain at least one finite value.")
    return sorted(set(values))


def _softmax_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(np.clip(shifted, -50.0, 50.0))
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def _apply_topk_pairwise_reranker(probabilities: np.ndarray, reranker: TopKPairwiseReranker | None) -> np.ndarray:
    """Apply a source-fitted one-vs-one reranker to only the top-k classes.

    The reranker never sees the held-out subject during fitting.  It only
    recalibrates ambiguous top-k decisions using source-subject OOF probability
    margins, so it is a decision-layer correction rather than an additional
    target-subject training step.
    """

    probabilities = np.asarray(probabilities, dtype=float)
    if reranker is None or reranker.alpha <= 0.0 or reranker.top_k <= 1:
        return probabilities
    n_classes = int(reranker.n_classes)
    if probabilities.ndim != 2 or probabilities.shape[1] != n_classes:
        raise ValueError("Reranker class count does not match probability matrix.")
    intercepts = np.asarray(reranker.intercepts, dtype=float).reshape(n_classes, n_classes)
    slopes = np.asarray(reranker.slopes, dtype=float).reshape(n_classes, n_classes)
    log_probabilities = np.log(np.clip(probabilities, DEFAULT_STACKING_EPSILON, 1.0))
    scores = log_probabilities.copy()
    top_k = min(int(reranker.top_k), n_classes)
    top_indices = np.argsort(log_probabilities, axis=1)[:, -top_k:]
    for row_index, row_classes in enumerate(top_indices):
        bonuses = np.zeros(n_classes, dtype=float)
        for left_position in range(len(row_classes)):
            for right_position in range(left_position + 1, len(row_classes)):
                left = int(row_classes[left_position])
                right = int(row_classes[right_position])
                class_i, class_j = (left, right) if left < right else (right, left)
                margin = log_probabilities[row_index, class_i] - log_probabilities[row_index, class_j]
                pairwise_logit = intercepts[class_i, class_j] + slopes[class_i, class_j] * margin
                bonuses[class_i] += pairwise_logit
                bonuses[class_j] -= pairwise_logit
        scores[row_index, row_classes] += float(reranker.alpha) * bonuses[row_classes] / float(max(top_k - 1, 1))
    return _softmax_scores(scores)


def _fit_topk_pairwise_reranker(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    n_classes: int,
    top_k: int,
    alpha_grid: Sequence[float] | None = None,
) -> TopKPairwiseReranker | None:
    """Fit a leakage-safe top-k pairwise reranker from source OOF probabilities."""

    top_k = min(_normalize_rerank_top_k(top_k), int(n_classes))
    if top_k <= 1:
        return None
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if probabilities.shape != (labels.shape[0], int(n_classes)):
        raise ValueError("Reranker probabilities must have shape (n_samples, n_classes).")
    log_probabilities = np.log(np.clip(probabilities, DEFAULT_STACKING_EPSILON, 1.0))
    intercepts = np.zeros((int(n_classes), int(n_classes)), dtype=float)
    slopes = np.zeros((int(n_classes), int(n_classes)), dtype=float)
    for class_i in range(int(n_classes)):
        for class_j in range(class_i + 1, int(n_classes)):
            mask = (labels == class_i) | (labels == class_j)
            if np.unique(labels[mask]).size < 2:
                continue
            margin = (log_probabilities[mask, class_i] - log_probabilities[mask, class_j]).reshape(-1, 1)
            target = (labels[mask] == class_i).astype(int)
            if float(np.std(margin)) <= 1e-12:
                continue
            try:
                model = LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=200)
                model.fit(margin, target)
            except ValueError:
                continue
            intercepts[class_i, class_j] = float(model.intercept_[0])
            slopes[class_i, class_j] = float(model.coef_[0, 0])

    alpha_values = _parse_float_grid(alpha_grid, DEFAULT_RERANK_ALPHA_GRID)
    if 0.0 not in alpha_values:
        alpha_values = [0.0, *alpha_values]
    best_alpha = 0.0
    best_score = _candidate_metrics(probabilities, labels, n_classes=int(n_classes))["balanced_accuracy"]
    for alpha in alpha_values:
        candidate = TopKPairwiseReranker(
            n_classes=int(n_classes),
            top_k=top_k,
            alpha=float(alpha),
            intercepts=tuple(float(value) for value in intercepts.ravel()),
            slopes=tuple(float(value) for value in slopes.ravel()),
        )
        score = _candidate_metrics(_apply_topk_pairwise_reranker(probabilities, candidate), labels, n_classes=int(n_classes))["balanced_accuracy"]
        if score > best_score + 1e-12:
            best_score = score
            best_alpha = float(alpha)
    if best_alpha <= 0.0:
        return None
    return TopKPairwiseReranker(
        n_classes=int(n_classes),
        top_k=top_k,
        alpha=best_alpha,
        intercepts=tuple(float(value) for value in intercepts.ravel()),
        slopes=tuple(float(value) for value in slopes.ravel()),
    )


def _weights_from_scores(scores: Sequence[float], *, weighting: str, temperature: float | None) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or scores.size == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("Ensemble scores must be a non-empty finite vector.")
    weighting = _normalize_weighting(weighting)
    if weighting in {"uniform", "stacked"} or scores.size == 1:
        weights = np.ones(scores.size, dtype=float)
    elif weighting == "rank":
        weights = 1.0 / np.arange(1, scores.size + 1, dtype=float)
    else:
        temperature = _normalize_temperature(temperature, weighting)
        weights = np.exp(np.clip((scores - float(np.max(scores))) / float(temperature), -60.0, 0.0))
    return weights / float(np.sum(weights))


def _candidate_summary(candidate: CandidateSpec, rows: Sequence[Mapping[str, Any]], metric: str) -> CandidateSummary:
    frame = pd.DataFrame(rows)
    mean_score = float(frame[metric].mean())
    std_score = float(frame[metric].std(ddof=0))
    return CandidateSummary(candidate, mean_score, std_score, _larger_is_better(mean_score, metric), len(frame))


def _class_balanced_sample_weights(labels: np.ndarray, *, n_classes: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if labels.size == 0:
        raise ValueError("Need at least one source prediction for ensemble stacking.")
    counts = np.bincount(labels, minlength=int(n_classes)).astype(float)
    weights = np.zeros(labels.shape[0], dtype=float)
    observed = counts[labels] > 0.0
    weights[observed] = 1.0 / counts[labels[observed]]
    if float(weights.mean()) <= 0.0:
        raise ValueError("Cannot compute class-balanced stacking weights from empty labels.")
    return weights / float(weights.mean())


def _fit_stacking_weights(
    probability_cube: np.ndarray,
    labels: np.ndarray,
    *,
    n_classes: int,
    max_iter: int = DEFAULT_STACKING_MAX_ITER,
    learning_rate: float = DEFAULT_STACKING_LEARNING_RATE,
    epsilon: float = DEFAULT_STACKING_EPSILON,
) -> np.ndarray:
    """Fit non-negative ensemble weights from source-only out-of-fold probabilities.

    The objective is class-balanced log loss on the inner source-subject LOSO
    predictions.  Only source-subject predictions are used, so this is a
    leakage-safe stacking layer rather than a held-out-subject calibration step.
    """

    cube = np.asarray(probability_cube, dtype=float)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if cube.ndim != 3:
        raise ValueError("probability_cube must have shape (n_candidates, n_samples, n_classes).")
    n_candidates, n_samples, cube_classes = cube.shape
    if n_candidates < 1 or n_samples != labels.shape[0] or cube_classes != int(n_classes):
        raise ValueError("probability_cube shape is inconsistent with labels or n_classes.")
    if n_candidates == 1:
        return np.ones(1, dtype=float)
    cube = np.clip(cube, float(epsilon), 1.0)
    cube /= cube.sum(axis=2, keepdims=True)
    true_probabilities = cube[:, np.arange(n_samples), labels]
    sample_weights = _class_balanced_sample_weights(labels, n_classes=n_classes)
    weights = np.full(n_candidates, 1.0 / float(n_candidates), dtype=float)
    max_iter = max(1, int(max_iter))
    learning_rate = float(learning_rate)
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("Stacking learning_rate must be positive and finite.")

    for iteration in range(max_iter):
        denominator = np.clip(weights @ true_probabilities, float(epsilon), 1.0)
        gradient = -np.average(true_probabilities / denominator[None, :], axis=1, weights=sample_weights)
        gradient -= float(np.dot(gradient, weights))
        step = learning_rate / np.sqrt(float(iteration + 1))
        updated = weights * np.exp(np.clip(-step * gradient, -50.0, 50.0))
        total = float(updated.sum())
        if not np.isfinite(total) or total <= 0.0:
            break
        weights = updated / total
    return weights / float(weights.sum())


def _source_oof_probabilities(
    *,
    subjects: Mapping[str, SubjectEpochs],
    cache: FeatureCache,
    candidates: Sequence[CandidateSpec],
    outer_test_subject: str,
    n_classes: int,
    max_iter: int,
    cue_source_weights: CueSourceWeights | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return candidate × source-trial × class probabilities from inner LOSO."""

    source_subjects = [subject for subject in sorted(subjects) if subject != outer_test_subject]
    if not candidates:
        raise ValueError("At least one selected candidate is required for source OOF stacking.")
    probability_blocks: list[list[np.ndarray]] = [[] for _ in candidates]
    label_blocks: list[np.ndarray] = []
    for inner_test_subject in source_subjects:
        train_subjects = [subject for subject in source_subjects if subject != inner_test_subject]
        subject_weights = cue_source_weights_from_summaries(
            cue_summaries or {},
            test_subject=inner_test_subject,
            train_subjects=train_subjects,
            mode=cue_source_weighting,
            temperature=cue_source_temperature,
        )
        label_blocks.append(subjects[inner_test_subject].labels)
        for candidate_index, candidate in enumerate(candidates):
            probability_blocks[candidate_index].append(
                _predict_candidate(
                    subjects=subjects,
                    cache=cache,
                    candidate=candidate,
                    train_subjects=train_subjects,
                    test_subject=inner_test_subject,
                    n_classes=n_classes,
                    max_iter=max_iter,
                    subject_weight_multipliers=None if cue_source_weights is None else cue_source_weights.for_fold(inner_test_subject, train_subjects),
                )
            )
    probability_cube = np.stack([np.vstack(blocks) for blocks in probability_blocks], axis=0)
    labels = np.concatenate(label_blocks, axis=0)
    return probability_cube, labels


def _calibrate_ensemble_from_oof(
    *,
    subjects: Mapping[str, SubjectEpochs],
    cache: FeatureCache,
    selected: Sequence[CandidateSummary],
    initial_weights: np.ndarray,
    outer_test_subject: str,
    n_classes: int,
    max_iter: int,
    weighting: str,
    class_bias: str,
    cue_source_weights: CueSourceWeights | None = None,
    rerank_top_k: int = DEFAULT_RERANK_TOP_K,
    rerank_alpha_grid: Sequence[float] | None = None,
) -> tuple[np.ndarray, tuple[float, ...], TopKPairwiseReranker | None, dict[str, float]]:
    weights = np.asarray(initial_weights, dtype=float).reshape(-1)
    if weighting != "stacked" and class_bias == "none" and _normalize_rerank_top_k(rerank_top_k) <= 1:
        return weights, (), None, {}

    probability_cube, labels = _source_oof_probabilities(
        subjects=subjects,
        cache=cache,
        candidates=[item.candidate for item in selected],
        outer_test_subject=outer_test_subject,
        n_classes=n_classes,
        max_iter=max_iter,
        cue_source_weights=cue_source_weights,
    )
    if weighting == "stacked":
        weights = _fit_stacking_weights(probability_cube, labels, n_classes=n_classes)
    combined = _renormalize(np.tensordot(weights, probability_cube, axes=(0, 0)))
    bias = _fit_class_bias(combined, labels, n_classes=n_classes, mode=class_bias)
    if class_bias != "none":
        combined = _apply_class_bias(combined, bias)
    reranker = _fit_topk_pairwise_reranker(
        combined,
        labels,
        n_classes=n_classes,
        top_k=rerank_top_k,
        alpha_grid=rerank_alpha_grid,
    )
    combined = _apply_topk_pairwise_reranker(combined, reranker)
    metrics = _candidate_metrics(combined, labels, n_classes=n_classes)
    bias_tuple = tuple(float(value) for value in bias) if class_bias != "none" else ()
    return weights, bias_tuple, reranker, metrics


def _select_ensemble(
    *,
    subjects: Mapping[str, SubjectEpochs],
    cache: FeatureCache,
    candidates: Sequence[CandidateSpec],
    outer_test_subject: str,
    n_classes: int,
    max_iter: int,
    metric: str,
    top_k: int,
    weighting: str,
    temperature: float | None,
    class_bias: str = DEFAULT_ENSEMBLE_CLASS_BIAS,
    cue_source_weights: CueSourceWeights | None = None,
    rerank_top_k: int = DEFAULT_RERANK_TOP_K,
    rerank_alpha_grid: Sequence[float] | None = None,
) -> tuple[EnsembleSelection, list[dict[str, Any]], list[dict[str, Any]]]:
    if metric not in SUPPORTED_SELECTION_METRICS:
        raise ValueError(f"Unknown selection metric {metric!r}; choose one of {sorted(SUPPORTED_SELECTION_METRICS)}.")
    if top_k < 1:
        raise ValueError("source_loso.ensemble_top_k must be at least 1.")
    inner_rows: list[dict[str, Any]] = []
    summaries: list[CandidateSummary] = []
    for candidate in candidates:
        rows = _inner_loso_scores(
            subjects=subjects,
            cache=cache,
            candidate=candidate,
            outer_test_subject=outer_test_subject,
            n_classes=n_classes,
            max_iter=max_iter,
            cue_source_weights=cue_source_weights,
        )
        inner_rows.extend(rows)
        summaries.append(_candidate_summary(candidate, rows, metric))
    ranked = sorted(summaries, key=lambda item: (-item.comparable_score, item.candidate.name))
    selected = ranked[: min(top_k, len(ranked))]
    weights = _weights_from_scores([item.comparable_score for item in selected], weighting=weighting, temperature=temperature)
    class_bias = _normalize_ensemble_class_bias(class_bias)
    weights, bias, reranker, oof_metrics = _calibrate_ensemble_from_oof(
        subjects=subjects,
        cache=cache,
        selected=selected,
        initial_weights=weights,
        outer_test_subject=outer_test_subject,
        n_classes=n_classes,
        max_iter=max_iter,
        weighting=weighting,
        class_bias=class_bias,
        cue_source_weights=cue_source_weights,
        rerank_top_k=rerank_top_k,
        rerank_alpha_grid=rerank_alpha_grid,
    )
    members = tuple(EnsembleMember(item.candidate, rank, float(weight), item.mean_score, item.std_score, item.comparable_score) for rank, (item, weight) in enumerate(zip(selected, weights, strict=True), start=1))
    selection = EnsembleSelection(
        members,
        metric,
        weighting,
        temperature,
        class_bias_mode=class_bias,
        class_bias=bias,
        reranker=reranker,
        oof_balanced_accuracy=oof_metrics.get("balanced_accuracy"),
        oof_log_loss=oof_metrics.get("log_loss"),
    )
    rank_rows = [
        {
            "outer_test_subject": outer_test_subject,
            "rank": rank,
            "selected": rank <= len(members),
            **_candidate_rowspec(item.candidate),
            "inner_selection_metric": metric,
            "inner_mean_score": item.mean_score,
            "inner_std_score": item.std_score,
            "inner_comparable_score": item.comparable_score,
            "inner_n_folds": item.n_folds,
            "ensemble_weight": float(weights[rank - 1]) if rank <= len(weights) else 0.0,
            "ensemble_class_bias": class_bias,
            "ensemble_oof_balanced_accuracy": "" if selection.oof_balanced_accuracy is None else selection.oof_balanced_accuracy,
            "ensemble_rerank_top_k": "" if selection.reranker is None else selection.reranker.top_k,
            "ensemble_rerank_alpha": "" if selection.reranker is None else selection.reranker.alpha,
            "ensemble_oof_log_loss": "" if selection.oof_log_loss is None else selection.oof_log_loss,
        }
        for rank, item in enumerate(ranked, start=1)
    ]
    return selection, inner_rows, rank_rows


def _renormalize(probabilities: np.ndarray) -> np.ndarray:
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0) or not np.all(np.isfinite(row_sums)):
        raise ValueError("Ensemble probabilities must have positive finite row sums.")
    return probabilities / row_sums


def _predict_ensemble(
    *,
    subjects: Mapping[str, SubjectEpochs],
    cache: FeatureCache,
    selection: EnsembleSelection,
    train_subjects: Sequence[str],
    test_subject: str,
    n_classes: int,
    max_iter: int,
    subject_weight_multipliers: Mapping[str, float] | None = None,
) -> np.ndarray:
    probability_sum = np.zeros((len(subjects[test_subject].labels), n_classes), dtype=float)
    for member in selection.members:
        probability_sum += member.weight * _predict_candidate(subjects=subjects, cache=cache, candidate=member.candidate, train_subjects=train_subjects, test_subject=test_subject, n_classes=n_classes, max_iter=max_iter, subject_weight_multipliers=subject_weight_multipliers)
    combined = _renormalize(probability_sum)
    if selection.class_bias:
        combined = _apply_class_bias(combined, np.asarray(selection.class_bias, dtype=float))
    if selection.reranker is not None:
        combined = _apply_topk_pairwise_reranker(combined, selection.reranker)
    return combined


def _selection_rowspec(selection: EnsembleSelection) -> dict[str, Any]:
    primary = selection.members[0]
    return {
        **_candidate_rowspec(primary.candidate),
        "candidate": selection.name,
        "primary_candidate": primary.candidate.name,
        "ensemble_size": len(selection.members),
        "ensemble_weighting": selection.weighting,
        "ensemble_temperature": "" if selection.temperature is None else selection.temperature,
        "ensemble_class_bias": selection.class_bias_mode,
        "ensemble_class_bias_values": "|".join(f"{value:.8g}" for value in selection.class_bias),
        "ensemble_rerank_top_k": "" if selection.reranker is None else selection.reranker.top_k,
        "ensemble_rerank_alpha": "" if selection.reranker is None else selection.reranker.alpha,
        "ensemble_oof_balanced_accuracy": "" if selection.oof_balanced_accuracy is None else selection.oof_balanced_accuracy,
        "ensemble_oof_log_loss": "" if selection.oof_log_loss is None else selection.oof_log_loss,
        "ensemble_candidates": "|".join(member.candidate.name for member in selection.members),
        "ensemble_weights": "|".join(f"{member.weight:.8g}" for member in selection.members),
        "ensemble_inner_scores": "|".join(f"{member.mean_score:.8g}" for member in selection.members),
        "ensemble_decoders": "|".join(dict.fromkeys(str(member.candidate.decoder) for member in selection.members)),
    }


def _write_json_sidecar(path: Path, payload: Mapping[str, Any]) -> None:
    Path(str(path) + ".provenance.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def run_bushmeg_source_loso_ensemble(
    config_path: str | Path,
    *,
    overrides: Sequence[str] | None = None,
    out_path: str | Path | None = None,
    inner_cv_out_path: str | Path | None = None,
    predictions_out_path: str | Path | None = None,
    candidate_summary_out_path: str | Path | None = None,
) -> pd.DataFrame:
    config_path = Path(config_path)
    config = apply_overrides(load_config(config_path), overrides)
    source_loso = _section(config, "source_loso")
    metric = str(source_loso.get("selection_metric", DEFAULT_SELECTION_METRIC))
    top_k = int(source_loso.get("ensemble_top_k", DEFAULT_ENSEMBLE_TOP_K))
    weighting = _normalize_weighting(source_loso.get("ensemble_weighting", DEFAULT_ENSEMBLE_WEIGHTING))
    temperature = _normalize_temperature(source_loso.get("ensemble_temperature", DEFAULT_ENSEMBLE_TEMPERATURE), weighting)
    class_bias = _normalize_ensemble_class_bias(source_loso.get("ensemble_class_bias", DEFAULT_ENSEMBLE_CLASS_BIAS))
    rerank_top_k = _normalize_rerank_top_k(source_loso.get("rerank_top_k", source_loso.get("ensemble_rerank_top_k", DEFAULT_RERANK_TOP_K)))
    rerank_alpha_grid = _parse_float_grid(source_loso.get("rerank_alpha_grid", source_loso.get("ensemble_rerank_alpha_grid", DEFAULT_RERANK_ALPHA_GRID)), DEFAULT_RERANK_ALPHA_GRID)
    max_iter = int((_section(config, "decoding") or {}).get("max_iter", 1000))

    subjects, encoder = _load_subjects_from_config(config, config_dir=config_path.parent)
    candidates = _candidate_grid(config)
    cache = FeatureCache(subjects)
    cue_source_weights = resolve_cue_source_weights(config, config_dir=config_path.parent, known_subjects=subjects)
    n_classes = len(encoder.classes_)

    out = Path(out_path) if out_path is not None else _resolve_output(config, config_dir=config_path.parent, key="source_loso_ensemble_summary_csv", default="source_loso_ensemble_summary.csv")
    inner_out = Path(inner_cv_out_path) if inner_cv_out_path is not None else _resolve_output(config, config_dir=config_path.parent, key="source_loso_ensemble_inner_cv_csv", default="source_loso_ensemble_inner_cv.csv")
    pred_out = Path(predictions_out_path) if predictions_out_path is not None else _resolve_output(config, config_dir=config_path.parent, key="source_loso_ensemble_predictions_csv", default="source_loso_ensemble_predictions.csv")
    rank_out = Path(candidate_summary_out_path) if candidate_summary_out_path is not None else _resolve_output(config, config_dir=config_path.parent, key="source_loso_ensemble_candidate_summary_csv", default="source_loso_ensemble_candidate_summary.csv")
    cue_weights_out = _resolve_output(config, config_dir=config_path.parent, key="source_loso_ensemble_cue_weights_csv", default="source_loso_ensemble_cue_source_weights.csv")

    summary_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for outer_test_subject in sorted(subjects):
        selection, fold_inner_rows, fold_rank_rows = _select_ensemble(
            subjects=subjects,
            cache=cache,
            candidates=candidates,
            outer_test_subject=outer_test_subject,
            n_classes=n_classes,
            max_iter=max_iter,
            metric=metric,
            top_k=top_k,
            weighting=weighting,
            temperature=temperature,
            class_bias=class_bias,
            cue_source_weights=cue_source_weights,
            rerank_top_k=rerank_top_k,
            rerank_alpha_grid=rerank_alpha_grid,
        )
        inner_rows.extend(fold_inner_rows)
        rank_rows.extend(fold_rank_rows)
        train_subjects = [subject for subject in sorted(subjects) if subject != outer_test_subject]
        fold_subject_weights = None if cue_source_weights is None else cue_source_weights.for_fold(outer_test_subject, train_subjects)
        probabilities = _predict_ensemble(subjects=subjects, cache=cache, selection=selection, train_subjects=train_subjects, test_subject=outer_test_subject, n_classes=n_classes, max_iter=max_iter, subject_weight_multipliers=fold_subject_weights)
        labels = subjects[outer_test_subject].labels
        predictions = probabilities.argmax(axis=1)
        member_scores = np.asarray([member.mean_score for member in selection.members], dtype=float)
        member_weights = np.asarray([member.weight for member in selection.members], dtype=float)
        summary_rows.append({
            "outer_test_subject": outer_test_subject,
            **_selection_rowspec(selection),
            "inner_selection_metric": metric,
            "inner_mean_score": float(np.average(member_scores, weights=member_weights)),
            "inner_best_score": float(np.min(member_scores) if metric in MINIMIZE_SELECTION_METRICS else np.max(member_scores)),
            "inner_std_score": float(np.std(member_scores, ddof=0)),
            "inner_n_folds": len(train_subjects),
            "cue_source_weighting": "" if cue_source_weights is None else cue_source_weights.mode,
            "cue_source_weighting_blend": "" if cue_source_weights is None else cue_source_weights.blend,
            "cue_source_weights": "" if not fold_subject_weights else "|".join(f"{subject}:{weight:.8g}" for subject, weight in sorted(fold_subject_weights.items())),
            **_candidate_metrics(probabilities, labels, n_classes=n_classes),
            "n_train_subjects": len(train_subjects),
            "n_test_trials": len(labels),
            "n_classes": n_classes,
            "class_names": "|".join(map(str, encoder.classes_)),
        })
        metadata = subjects[outer_test_subject].metadata.reset_index(drop=True)
        for row_idx, (true_label, predicted_label) in enumerate(zip(labels, predictions, strict=True)):
            row: dict[str, Any] = {
                "outer_test_subject": outer_test_subject,
                "trial_index": int(row_idx),
                "candidate": selection.name,
                "primary_candidate": selection.members[0].candidate.name,
                "ensemble_size": len(selection.members),
                "ensemble_candidates": "|".join(member.candidate.name for member in selection.members),
                "ensemble_weights": "|".join(f"{member.weight:.8g}" for member in selection.members),
                "ensemble_class_bias": selection.class_bias_mode,
                "true_label": int(true_label),
                "true_class": str(encoder.classes_[true_label]),
                "predicted_label": int(predicted_label),
                "predicted_class": str(encoder.classes_[predicted_label]),
                "probability_true_class": float(probabilities[row_idx, true_label]),
                "confidence": float(np.max(probabilities[row_idx])),
                "is_correct": bool(predicted_label == true_label),
            }
            for column in ("participant", "condition", "stimulus_class"):
                if column in metadata.columns:
                    row[column] = metadata.loc[row_idx, column]
            for class_idx, class_name in enumerate(encoder.classes_):
                row[f"class_{class_idx}"] = str(class_name)
                row[f"prob_class_{class_idx}"] = float(probabilities[row_idx, class_idx])
            prediction_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    for path, frame in ((out, summary), (inner_out, pd.DataFrame(inner_rows)), (rank_out, pd.DataFrame(rank_rows)), (pred_out, pd.DataFrame(prediction_rows))):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    if cue_source_weights is not None:
        write_cue_source_weight_csv(cue_source_weights, sorted(subjects), cue_weights_out)
    _write_json_sidecar(
        out,
        {
            "config_path": str(config_path),
            "selection_metric": metric,
            "ensemble_top_k": top_k,
            "ensemble_weighting": weighting,
            "ensemble_temperature": temperature,
            "ensemble_class_bias": class_bias,
            "rerank_top_k": rerank_top_k,
            "rerank_alpha_grid": rerank_alpha_grid,
            "stacking_max_iter": DEFAULT_STACKING_MAX_ITER,
            "n_subjects": len(subjects),
            "n_candidates": len(candidates),
            "cue_files_used": cue_source_weights is not None,
            "cue_files_used_for_classifier_training": False,
            "cue_source_weighting_config": {} if cue_source_weights is None else dict(cue_source_weights.config or {}),
            "target_labels_used_for_selection": False,
            "target_unlabeled_data_used_for_calibration": False,
            "random_seed": DEFAULT_RANDOM_SEED,
        },
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run cue-free BUSH-MEG source-only nested top-k ensemble LOSO decoding.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--inner-cv-out", type=Path)
    parser.add_argument("--candidate-summary-out", type=Path)
    parser.add_argument("--predictions-out", type=Path)
    args = parser.parse_args(argv)
    summary = run_bushmeg_source_loso_ensemble(args.config, overrides=args.overrides, out_path=args.out, inner_cv_out_path=args.inner_cv_out, candidate_summary_out_path=args.candidate_summary_out, predictions_out_path=args.predictions_out)
    print(f"Wrote {len(summary)} LOSO rows")
    print(f"Mean balanced accuracy: {float(summary['balanced_accuracy'].mean()):.6f}")
    print(f"Mean top-2/top-3 accuracy: {float(summary['top2_accuracy'].mean()):.6f} / {float(summary['top3_accuracy'].mean()):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
