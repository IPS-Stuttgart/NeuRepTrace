"""Resolve all-protocol prediction metric edge cases."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

_PATCH_MARKER = "_neureptrace_bushmeg_all_protocols_topk_tie_patch_installed"
_LABEL_INDEX_ERROR = "labels must contain finite integer class indices."


def _normalize_k(k: Any) -> int:
    if isinstance(k, (bool, np.bool_)):
        raise ValueError("k must be a positive integer.")
    try:
        k_float = float(k)
    except (TypeError, ValueError) as exc:
        raise ValueError("k must be a positive integer.") from exc
    if not np.isfinite(k_float) or k_float % 1.0 != 0.0 or k_float < 1.0:
        raise ValueError("k must be a positive integer.")
    return int(k_float)


def _contains_boolean_token(values: Any) -> bool:
    try:
        array = np.asarray(values, dtype=object)
    except (TypeError, ValueError):
        return isinstance(values, (bool, np.bool_))
    if array.ndim == 0:
        return isinstance(array.item(), (bool, np.bool_))
    return any(isinstance(value, (bool, np.bool_)) for value in array.reshape(-1).tolist())


def _label_index_vector(labels: Any) -> np.ndarray:
    if _contains_boolean_token(labels):
        raise ValueError(_LABEL_INDEX_ERROR)
    try:
        raw = np.asarray(labels)
    except (TypeError, ValueError) as exc:
        raise ValueError(_LABEL_INDEX_ERROR) from exc
    if raw.ndim == 0:
        raw = raw.reshape(1)
    elif raw.ndim == 2 and raw.shape[1] == 1:
        raw = raw.reshape(-1)
    elif raw.ndim != 1:
        raise ValueError("labels must have one entry per probability row.")
    try:
        numeric = raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(_LABEL_INDEX_ERROR) from exc
    if not np.all(np.isfinite(numeric)):
        raise ValueError(_LABEL_INDEX_ERROR)
    rounded = np.rint(numeric)
    if not np.all(np.isclose(numeric, rounded, rtol=0.0, atol=1.0e-12)):
        raise ValueError(_LABEL_INDEX_ERROR)
    return rounded.astype(int, copy=False)


def _top_k_accuracy(probabilities: np.ndarray, labels: np.ndarray, *, k: int) -> float:
    """Compute exact-k top-k accuracy with stable class-index tie handling."""

    probability_matrix = np.asarray(probabilities, dtype=float)
    label_indices = _label_index_vector(labels)
    if probability_matrix.size == 0:
        return float("nan")
    if probability_matrix.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional matrix.")
    if label_indices.size != probability_matrix.shape[0]:
        raise ValueError("labels must have one entry per probability row.")
    if probability_matrix.shape[1] == 0:
        return float("nan")

    k_value = min(_normalize_k(k), probability_matrix.shape[1])
    if k_value >= probability_matrix.shape[1]:
        valid_labels = (0 <= label_indices) & (label_indices < probability_matrix.shape[1])
        return float(np.mean(valid_labels))

    top_k = np.argsort(-probability_matrix, axis=1, kind="mergesort")[:, :k_value]
    hits = np.any(top_k == label_indices[:, None], axis=1)
    return float(np.mean(hits))


def _labels_to_probability_positions(group: pd.DataFrame, prob_columns: Sequence[str]) -> np.ndarray:
    """Map true labels to probability columns, preferring explicit label indices.

    Prediction tables may carry both a human/dataset-facing ``true_label`` and a
    model-facing ``true_label_index``.  When raw labels are numeric but not the
    probability-column indices, resolving ``true_label`` first silently assigns
    rows to the wrong probability column.  The explicit index column is the
    canonical metric key whenever it is present; named labels remain supported as
    a fallback through class_<index> metadata.
    """

    metric_patch = importlib.import_module("neureptrace._bushmeg_all_protocols_prediction_metric_patch")
    lookup = metric_patch._probability_position_lookup(prob_columns)
    class_index_by_name = metric_patch._class_name_index_map(group)
    class_name_by_index = {index: name for name, index in class_index_by_name.items()}
    label_columns = [column for column in ("true_label_index", "true_label") if column in group.columns]
    if not label_columns:
        raise ValueError("Prediction metrics require true_label or true_label_index.")

    positions: list[int] = []
    missing: list[Any] = []
    for _, row in group.iterrows():
        resolved: int | None = None
        for column in label_columns:
            value = row[column]
            resolved = metric_patch._resolve_probability_position(value, lookup, class_name_by_index)
            if resolved is not None:
                break
            class_index = class_index_by_name.get(str(value))
            if class_index is not None:
                resolved = metric_patch._resolve_probability_position(class_index, lookup, class_name_by_index)
                if resolved is not None:
                    break
        if resolved is None:
            missing.append(row[label_columns[0]])
        else:
            positions.append(resolved)
    if missing:
        preview = ", ".join(repr(value) for value in missing[:5])
        raise ValueError(f"Prediction metrics cannot map true label(s) to probability columns: {preview}.")
    return np.asarray(positions, dtype=int)


def _label_only_metric_vectors(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return comparable true/predicted vectors when probabilities are absent.

    The label-only path must mirror the probability path: explicit
    ``*_label_index`` columns are canonical when both true and predicted indices
    are available. Raw dataset labels can be numeric without being zero-based
    class positions, so resolving them before explicit indices corrupts metrics.
    """

    metric_patch = importlib.import_module("neureptrace._bushmeg_all_protocols_prediction_metric_patch")
    if "true_label_index" in group.columns and "predicted_label_index" in group.columns:
        true_indices = metric_patch._numeric_label_indices(group["true_label_index"])
        predicted_indices = metric_patch._numeric_label_indices(group["predicted_label_index"])
        if true_indices is not None and predicted_indices is not None:
            if true_indices.shape[0] != predicted_indices.shape[0]:
                raise ValueError("Prediction metrics require true and predicted labels to have the same row count.")
            return true_indices, predicted_indices

    true_indices = None
    predicted_indices = None
    if "true_label" in group.columns:
        true_indices = metric_patch._label_indices_from_named_column(group, "true_label")
    if true_indices is None and "true_label_index" in group.columns:
        true_indices = metric_patch._numeric_label_indices(group["true_label_index"])
    if "predicted_label" in group.columns:
        predicted_indices = metric_patch._label_indices_from_named_column(group, "predicted_label")
    if predicted_indices is None and "predicted_label_index" in group.columns:
        predicted_indices = metric_patch._numeric_label_indices(group["predicted_label_index"])

    if true_indices is not None and predicted_indices is not None:
        if true_indices.shape[0] != predicted_indices.shape[0]:
            raise ValueError("Prediction metrics require true and predicted labels to have the same row count.")
        return true_indices, predicted_indices

    true_values = metric_patch._raw_label_values(group, index_column="true_label_index", label_column="true_label", role="true")
    predicted_values = metric_patch._raw_label_values(
        group,
        index_column="predicted_label_index",
        label_column="predicted_label",
        role="predicted",
    )
    if true_values.shape[0] != predicted_values.shape[0]:
        raise ValueError("Prediction metrics require true and predicted labels to have the same row count.")
    return true_values.astype(str), predicted_values.astype(str)


def install() -> None:
    """Patch all-protocol prediction metric recomputation edge cases."""

    report_patch = importlib.import_module("neureptrace._bushmeg_all_protocols_report_protocol_labels_patch")
    report_patch.install()

    all_protocols = importlib.import_module("neureptrace.bushmeg_all_protocols")
    if getattr(all_protocols, _PATCH_MARKER, False):
        return

    metric_patch = importlib.import_module("neureptrace._bushmeg_all_protocols_prediction_metric_patch")
    metric_patch._top_k_accuracy = _top_k_accuracy
    metric_patch._labels_to_probability_positions = _labels_to_probability_positions
    metric_patch._label_only_metric_vectors = _label_only_metric_vectors
    all_protocols._top_k_accuracy = _top_k_accuracy
    all_protocols._labels_to_probability_positions = _labels_to_probability_positions
    all_protocols._label_only_metric_vectors = _label_only_metric_vectors
    setattr(all_protocols, _PATCH_MARKER, True)


__all__ = ["install"]