from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def rank_class_scores(
    scores: Sequence[Sequence[float]] | np.ndarray | None,
    classes: Sequence | np.ndarray | None,
    y_true: Sequence | np.ndarray,
    *,
    top_k: Sequence[int] = (2, 3),
    row_top_k: int = 3,
    class_column: str = "class",
) -> dict[str, object]:
    """Rank true labels in a per-class score matrix and compute top-k metrics.

    Missing true labels are counted as top-k failures but are excluded from the
    finite mean/median rank. If no class-score columns are available, top-k and
    rank summaries are undefined and returned as ``NaN``.
    """

    y_true = _label_vector(y_true, name="y_true")
    top_k = tuple(_validate_integer(k, name="top_k", minimum=1) for k in top_k)
    row_top_k = _validate_integer(row_top_k, name="row_top_k", minimum=0)
    if not class_column:
        raise ValueError("class_column must be non-empty.")

    if scores is None or classes is None:
        return _empty_class_rank_result(y_true, top_k)

    score_matrix = np.asarray(scores, dtype=float)
    class_order = _label_vector(classes, name="classes")
    if score_matrix.ndim != 2:
        raise ValueError("scores must be a two-dimensional matrix.")
    if score_matrix.shape[0] != y_true.shape[0]:
        raise ValueError("scores and y_true must contain the same samples.")
    if score_matrix.shape[1] != class_order.size:
        raise ValueError("scores columns must match classes.")
    if not np.all(np.isfinite(score_matrix)):
        raise ValueError("scores must contain only finite values.")
    duplicate_class = _find_duplicate_class_label(class_order)
    if duplicate_class is not None:
        raise ValueError(f"classes must be unique; duplicate label {duplicate_class!r} found.")
    if score_matrix.shape[1] == 0:
        return _empty_class_rank_result(y_true, top_k)

    order = np.argsort(-score_matrix, axis=1, kind="mergesort")
    top_hits = {k: [] for k in top_k}
    ranks: list[float] = []
    rows: list[dict[str, object]] = []
    for sample_index, truth in enumerate(y_true):
        ranked = class_order[order[sample_index]]
        match = _matching_class_positions(ranked, truth)
        for k in top_k:
            top_hits[k].append(bool(match.size and match[0] < k))
        rank = float(match[0] + 1) if match.size else np.nan
        ranks.append(rank)
        row: dict[str, object] = {"true_label_rank": rank, "true_label_score": np.nan}
        true_index = _matching_class_positions(class_order, truth)
        if true_index.size:
            row["true_label_score"] = float(score_matrix[sample_index, true_index[0]])
        for position, class_index in enumerate(order[sample_index, :row_top_k], start=1):
            row[f"rank{position}_{class_column}"] = _as_python_scalar(class_order[class_index])
            row[f"rank{position}_score"] = float(score_matrix[sample_index, class_index])
        rows.append(row)

    true_label_ranks = np.asarray(ranks, dtype=float)
    return {
        "top_k_accuracy": {k: float(np.mean(top_hits[k])) for k in top_k},
        "true_label_ranks": true_label_ranks,
        "mean_true_label_rank": _finite_nanmean(true_label_ranks),
        "median_true_label_rank": _finite_nanmedian(true_label_ranks),
        "rows": rows,
    }


def _label_vector(values: Sequence | np.ndarray, *, name: str) -> np.ndarray:
    if isinstance(values, np.ndarray):
        vector = values.astype(object, copy=False)
        if vector.ndim == 0:
            return vector.reshape(1)
        if vector.ndim == 1:
            return vector
        if vector.shape[1] == 1:
            return vector.reshape(-1)
        rows = [tuple(row.tolist()) for row in vector.reshape(vector.shape[0], -1)]
        return _object_vector(rows)

    if isinstance(values, (str, bytes)):
        return np.asarray([values], dtype=object)

    try:
        items = list(values)
    except TypeError:
        items = [values]

    if _contains_composite_label(items):
        vector = _object_vector(items)
    else:
        vector = np.asarray(items, dtype=object)
    if vector.ndim == 0:
        return vector.reshape(1)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return vector


def _contains_composite_label(items: Sequence[object]) -> bool:
    return any(_is_composite_label(item) for item in items)


def _is_composite_label(value: object) -> bool:
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, np.ndarray):
        return value.ndim != 0
    return isinstance(value, (tuple, list, dict))


def _object_vector(items: Sequence[object]) -> np.ndarray:
    vector = np.empty(len(items), dtype=object)
    vector[:] = items
    return vector


def _validate_integer(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} values must be integers.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} values must be integers.") from exc
    if not np.isfinite(numeric) or numeric % 1.0 != 0.0:
        raise ValueError(f"{name} values must be integers.")
    integer = int(numeric)
    if integer < int(minimum):
        qualifier = "positive" if int(minimum) == 1 else "non-negative"
        raise ValueError(f"{name} values must be {qualifier}.")
    return integer


def _empty_class_rank_result(y_true: np.ndarray, top_k: Sequence[int]) -> dict[str, object]:
    ranks = np.full(y_true.shape[0], np.nan, dtype=float)
    return {
        "top_k_accuracy": {k: np.nan for k in top_k},
        "true_label_ranks": ranks,
        "mean_true_label_rank": np.nan,
        "median_true_label_rank": np.nan,
        "rows": [{} for _ in ranks],
    }


def _finite_nanmean(values: Sequence[float] | np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else np.nan


def _finite_nanmedian(values: Sequence[float] | np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else np.nan


def _matching_class_positions(labels: np.ndarray, truth) -> np.ndarray:
    return np.asarray([index for index, label in enumerate(labels) if _class_labels_equal(label, truth)], dtype=int)


def _find_duplicate_class_label(class_order: np.ndarray):
    for index, label in enumerate(class_order):
        for previous_label in class_order[:index]:
            if _class_labels_equal(label, previous_label):
                return _as_python_scalar(label)
    return None


def _class_labels_equal(left, right) -> bool:
    left = _as_python_scalar(left)
    right = _as_python_scalar(right)
    try:
        comparison = left == right
    except (TypeError, ValueError):
        comparison = False
    if isinstance(comparison, np.ndarray):
        try:
            return bool(np.all(comparison))
        except (TypeError, ValueError):
            return False
    try:
        if bool(comparison):
            return True
    except (TypeError, ValueError):
        pass
    try:
        return bool(np.isscalar(left) and np.isscalar(right) and np.isnan(left) and np.isnan(right))
    except (TypeError, ValueError):
        return False


def _as_python_scalar(value):
    return value.item() if isinstance(value, np.generic) else value
