from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence, Set
from numbers import Integral

import numpy as np

from neureptrace._object_label_utils import values_equal


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

    classes = _materialize_reusable_label_input(classes)
    y_true = _materialize_reusable_label_input(y_true)

    if classes is not None and _has_incompatible_array_label_shape(y_true, classes):
        raise ValueError("y_true must be one-dimensional.")
    y_true = _label_vector(y_true, name="y_true")
    if isinstance(top_k, (str, bytes, bytearray, memoryview, Mapping, Set)):
        raise ValueError("top_k must be a sequence of integers.")
    try:
        top_k_values = tuple(top_k)
    except TypeError as exc:
        raise ValueError("top_k must be a sequence of integers.") from exc
    top_k = tuple(_validate_integer(k, name="top_k", minimum=1) for k in top_k_values)
    row_top_k = _validate_integer(row_top_k, name="row_top_k", minimum=0)
    class_column = _validate_class_column_name(class_column)

    if scores is None or classes is None:
        return _empty_class_rank_result(y_true, top_k)

    scores = _materialize_reusable_score_input(scores)
    if _scores_contain_boolean(scores):
        raise ValueError("scores must contain numeric score values, not boolean flags.")
    if _scores_contain_complex(scores):
        raise ValueError("scores must contain real-valued scores, not complex values.")
    try:
        score_matrix = np.asarray(scores, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("scores must be a two-dimensional matrix.") from exc
    if score_matrix.ndim != 2:
        raise ValueError("scores must be a two-dimensional matrix.")
    if _has_incompatible_class_matrix(classes, expected_n_classes=score_matrix.shape[1]):
        raise ValueError("classes must be one-dimensional.")
    class_order = _label_vector(classes, name="classes")
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
    row_top_k = min(row_top_k, score_matrix.shape[1])

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


def _has_incompatible_array_label_shape(y_true: object, classes: object) -> bool:
    if not _is_matrix_label_array(y_true):
        return False
    label_shape = tuple(y_true.shape[1:])
    if not _is_matrix_label_array(classes):
        return not _sequence_labels_match_shape(classes, label_shape)
    return label_shape != tuple(classes.shape[1:])


def _sequence_labels_match_shape(values: object, label_shape: tuple[int, ...]) -> bool:
    if isinstance(values, (str, bytes)):
        return False
    try:
        items = list(values)
    except TypeError:
        return False
    if not items:
        return True
    return all(_label_item_shape(item) == label_shape for item in items)


def _label_item_shape(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        return ()
    if isinstance(value, np.ndarray):
        return tuple(value.shape) if value.ndim != 0 else ()
    if isinstance(value, (list, tuple)):
        return tuple(np.asarray(value, dtype=object).shape)
    return ()


def _has_incompatible_class_matrix(classes: object, *, expected_n_classes: int) -> bool:
    return _is_matrix_label_array(classes) and classes.shape[0] != expected_n_classes


def _is_matrix_label_array(values: object) -> bool:
    return isinstance(values, np.ndarray) and values.ndim > 1


def _materialize_reusable_score_input(values: object) -> object:
    """Return score inputs that can be inspected more than once without data loss."""

    if values is None or isinstance(values, (str, bytes)):
        return values
    if isinstance(values, np.ndarray):
        if values.dtype != object:
            return values
        return _materialize_reusable_score_input(values.tolist())
    if hasattr(values, "__array__"):
        return values
    if not isinstance(values, Iterable):
        return values
    return [_materialize_reusable_score_input(value) for value in values]


def _scores_contain_boolean(values: object) -> bool:
    if isinstance(values, (bool, np.bool_)):
        return True
    if isinstance(values, np.ndarray):
        if np.issubdtype(values.dtype, np.bool_):
            return True
        if values.dtype == object:
            return any(_scores_contain_boolean(value) for value in values.ravel(order="C"))
        return False
    if hasattr(values, "__array__"):
        try:
            return _scores_contain_boolean(np.asarray(values, dtype=object))
        except (TypeError, ValueError):
            return False
    if isinstance(values, (str, bytes)):
        return False
    if not isinstance(values, Iterable):
        return False
    return any(_scores_contain_boolean(value) for value in values)


def _scores_contain_complex(values: object) -> bool:
    if isinstance(values, (complex, np.complexfloating)):
        return True
    if isinstance(values, np.ndarray):
        if np.issubdtype(values.dtype, np.complexfloating):
            return True
        if values.dtype == object:
            return any(_scores_contain_complex(value) for value in values.ravel(order="C"))
        return False
    if hasattr(values, "__array__"):
        try:
            return _scores_contain_complex(np.asarray(values, dtype=object))
        except (TypeError, ValueError):
            return False
    if isinstance(values, (str, bytes)):
        return False
    if not isinstance(values, Iterable):
        return False
    return any(_scores_contain_complex(value) for value in values)


def _materialize_nested_label(value: object) -> object:
    """Materialize one-pass iterables inside a single label value."""

    if isinstance(value, np.ndarray):
        if value.dtype != object:
            return value
        materialized = np.empty(value.shape, dtype=object)
        for index in np.ndindex(value.shape):
            materialized[index] = _materialize_nested_label(value[index])
        return materialized
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, list):
        return [_materialize_nested_label(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_materialize_nested_label(item) for item in value)
    if isinstance(value, dict):
        return {
            _materialize_nested_label(key): _materialize_nested_label(item)
            for key, item in value.items()
        }
    if not isinstance(value, Iterable):
        return value
    return tuple(_materialize_nested_label(item) for item in value)


def _materialize_reusable_label_input(values: object) -> object:
    """Return label inputs that can be inspected more than once without data loss."""

    if values is None or isinstance(values, (str, bytes)):
        return values
    if isinstance(values, np.ndarray):
        return _materialize_nested_label(values)
    try:
        return [_materialize_nested_label(value) for value in values]
    except TypeError:
        return values


def _label_vector(values: Sequence | np.ndarray, *, name: str) -> np.ndarray:
    if isinstance(values, np.ndarray):
        array = np.asarray(values)
        if array.ndim == 0:
            return _object_vector([array[()]])
        if array.ndim == 1:
            return _object_vector([array[index] for index in range(array.shape[0])])
        if array.ndim == 2 and array.shape[1] == 1:
            raise ValueError(f"{name} must be one-dimensional.")
        flat_rows = array.reshape(array.shape[0], -1)
        rows = [tuple(flat_rows[row, column] for column in range(flat_rows.shape[1])) for row in range(flat_rows.shape[0])]
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


def _validate_class_column_name(value: object) -> str:
    if not isinstance(value, str) or value == "":
        raise ValueError("class_column must be a non-empty string.")
    if value == "score":
        raise ValueError("class_column='score' conflicts with generated rank*_score columns.")
    return value


def _validate_integer(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, np.ndarray):
        raise ValueError(f"{name} values must be integers.")
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} values must be integers.")
    if isinstance(value, Integral):
        integer = int(value)
    else:
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
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
    return values_equal(left, right)


def _as_python_scalar(value):
    if isinstance(value, (np.datetime64, np.timedelta64)) and bool(np.isnat(value)):
        return value
    return value.item() if isinstance(value, np.generic) else value
