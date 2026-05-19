"""Stable public helpers for class-score extraction and true-label ranking.

Dataset-specific projects often need the same three operations after fitting a
windowed decoder: obtain a per-class score matrix, rank the true class in that
matrix, and expose compact top-k/rank summaries.  NeuRepTrace already owns the
low-level primitives in :mod:`neureptrace.decoding.class_scores` and
:mod:`neureptrace.metrics.ranking`; this module provides a small, stable facade
around those primitives so downstream packages do not need local compatibility
wrappers for common score/rank result shapes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from neureptrace.decoding.class_scores import class_score_matrix
from neureptrace.decoding.windowed import WindowedModelBundle, transform_window_features
from neureptrace.metrics.ranking import rank_class_scores


def score_model_classes(
    model: Any,
    features: Sequence[Sequence[float]] | np.ndarray,
    *,
    classes: Sequence | np.ndarray | None = None,
    fallback_labels: Sequence | np.ndarray | None = None,
    score_methods: Sequence[str] = ("decision_function", "predict_proba"),
    predict_fallback: bool = False,
    empty_on_missing: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return per-class scores and class order for a fitted classifier.

    Parameters
    ----------
    model:
        Fitted sklearn-style estimator.  ``decision_function`` is preferred over
        ``predict_proba`` by default, matching :func:`class_score_matrix`.
    features:
        Two-dimensional ``(n_samples, n_features)`` feature matrix.
    classes:
        Explicit class order.  When omitted, estimator ``classes_`` attributes
        are inspected and ``fallback_labels`` may be used as a last resort.
    fallback_labels:
        Labels used to infer class order when the estimator does not expose it.
    score_methods:
        Ordered scoring methods to try on ``model``.
    predict_fallback:
        If true, fall back to a one-hot prediction matrix for estimators without
        score-producing methods.
    empty_on_missing:
        If true, return ``(n_samples, 0)`` NaN scores and an empty class vector
        when no score matrix can be obtained.  This is useful for downstream
        code paths that require an array-shaped failure value.
    """

    feature_matrix = _feature_matrix(features)
    scores, class_order = class_score_matrix(
        model,
        feature_matrix,
        classes=classes,
        fallback_labels=fallback_labels,
        score_methods=score_methods,
        predict_fallback=predict_fallback,
    )
    if empty_on_missing and (scores is None or class_order is None):
        return _empty_class_scores(feature_matrix.shape[0])
    return scores, class_order


def score_window_classes(
    model_bundle: WindowedModelBundle,
    features: Sequence[Sequence[float]] | np.ndarray,
    *,
    score_methods: Sequence[str] = ("decision_function", "predict_proba"),
    predict_fallback: bool = False,
    empty_on_missing: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return per-class scores for a fitted windowed model bundle.

    The bundle's fold-local feature transform is applied before scoring, so the
    caller passes features in the original window feature space.
    """

    transformed_features = transform_window_features(model_bundle, features)
    return score_model_classes(
        model_bundle.model,
        transformed_features,
        fallback_labels=model_bundle.train_labels,
        score_methods=score_methods,
        predict_fallback=predict_fallback,
        empty_on_missing=empty_on_missing,
    )


def summarize_class_ranks(
    y_true: Sequence | np.ndarray,
    scores: Sequence[Sequence[float]] | np.ndarray | None,
    classes: Sequence | np.ndarray | None,
    *,
    top_k: Sequence[int] = (2, 3),
    row_top_k: int = 0,
    class_column: str = "class",
) -> dict[str, object]:
    """Rank true labels in a class-score matrix.

    This is a public-signature wrapper around
    :func:`neureptrace.metrics.ranking.rank_class_scores` with ``y_true`` first,
    matching the calling convention used by most evaluation code.
    """

    return rank_class_scores(scores, classes, y_true, top_k=top_k, row_top_k=row_top_k, class_column=class_column)


def true_label_ranks(
    y_true: Sequence | np.ndarray,
    scores: Sequence[Sequence[float]] | np.ndarray | None,
    classes: Sequence | np.ndarray | None,
) -> np.ndarray:
    """Return one-based ranks of each true label in descending score order."""

    summary = summarize_class_ranks(y_true, scores, classes, top_k=(), row_top_k=0)
    return np.asarray(summary["true_label_ranks"], dtype=float)


def topk_rank_metrics(
    y_true: Sequence | np.ndarray,
    scores: Sequence[Sequence[float]] | np.ndarray | None,
    classes: Sequence | np.ndarray | None,
    *,
    top_k: Sequence[int] = (2, 3),
    include_true_label_ranks: bool = False,
    include_median: bool = False,
) -> dict[str, object]:
    """Return compact ``top{k}_accuracy`` and mean-rank metrics.

    Missing true labels count as top-k failures but are excluded from finite
    mean/median rank summaries, following :func:`rank_class_scores` semantics.
    """

    normalized_top_k = tuple(int(k) for k in top_k)
    summary = summarize_class_ranks(y_true, scores, classes, top_k=normalized_top_k, row_top_k=0)
    top_k_accuracy = summary["top_k_accuracy"]
    metrics: dict[str, object] = {f"top{k}_accuracy": top_k_accuracy[k] for k in normalized_top_k}
    metrics["mean_true_label_rank"] = summary["mean_true_label_rank"]
    if include_median:
        metrics["median_true_label_rank"] = summary["median_true_label_rank"]
    if include_true_label_ranks:
        metrics["true_label_ranks"] = summary["true_label_ranks"]
    return metrics


def rank_summary_rows(
    y_true: Sequence | np.ndarray,
    scores: Sequence[Sequence[float]] | np.ndarray | None,
    classes: Sequence | np.ndarray | None,
    *,
    row_top_k: int = 3,
    class_column: str = "class",
) -> list[dict[str, object]]:
    """Return per-sample rank rows for diagnostics and CSV exports."""

    summary = summarize_class_ranks(y_true, scores, classes, top_k=(), row_top_k=row_top_k, class_column=class_column)
    return list(summary["rows"])


def _feature_matrix(features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("features must be a two-dimensional feature matrix.")
    return matrix


def _empty_class_scores(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    return np.full((int(n_samples), 0), np.nan, dtype=float), np.asarray([], dtype=int)
