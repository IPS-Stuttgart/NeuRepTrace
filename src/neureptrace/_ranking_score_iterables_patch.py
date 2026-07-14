from __future__ import annotations

import importlib
import sys
from collections.abc import Iterable, Sequence

import numpy as np

_ranking = importlib.import_module("neureptrace.metrics.ranking")
_ORIGINAL_RANK_CLASS_SCORES = _ranking.rank_class_scores
_PATCHED = False


_SCORE_BOOLEAN_ERROR = "scores must contain numeric score values, not boolean flags."
_CLASS_COLUMN_COLLISION_ERROR = "class_column='score' conflicts with generated rank*_score columns."


def _materialize_score_iterables(value: object) -> object:
    """Materialize generator-backed score rows before NumPy conversion."""

    if isinstance(value, np.ndarray):
        if value.dtype != object:
            return value
        return _materialize_score_iterables(value.tolist())
    if hasattr(value, "__array__"):
        return value
    if isinstance(value, (str, bytes)):
        return value
    if not isinstance(value, Iterable):
        return value
    return [_materialize_score_iterables(item) for item in value]


def _coerce_score_matrix(scores: object) -> np.ndarray:
    materialized = _materialize_score_iterables(scores)
    if _ranking._scores_contain_boolean(materialized):
        raise ValueError(_SCORE_BOOLEAN_ERROR)
    try:
        return np.asarray(materialized, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("scores must be a two-dimensional matrix.") from exc


def _rank_class_scores_with_score_iterables(
    scores: Sequence[Sequence[float]] | np.ndarray | None,
    classes: Sequence | np.ndarray | None,
    y_true: Sequence | np.ndarray,
    *,
    top_k: Sequence[int] = (2, 3),
    row_top_k: int = 3,
    class_column: str = "class",
) -> dict[str, object]:
    if isinstance(class_column, str) and class_column == "score":
        raise ValueError(_CLASS_COLUMN_COLLISION_ERROR)
    if scores is not None:
        scores = _coerce_score_matrix(scores)
    return _ORIGINAL_RANK_CLASS_SCORES(scores, classes, y_true, top_k=top_k, row_top_k=row_top_k, class_column=class_column)


def install() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _ranking.rank_class_scores = _rank_class_scores_with_score_iterables
    metrics_package = sys.modules.get("neureptrace.metrics")
    if metrics_package is not None:
        metrics_package.rank_class_scores = _rank_class_scores_with_score_iterables
    _PATCHED = True


__all__ = ["install"]
