"""Allow BUSH-MEG all-protocol metric recomputation for named labels."""

from __future__ import annotations

import importlib
import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss

from neureptrace.metrics import brier_score_multiclass, expected_calibration_error

_PATCH_MARKER = "_neureptrace_bushmeg_all_protocols_prediction_metric_patch_installed"
_CLASS_COLUMN_RE = re.compile(r"^class_(\d+)$")


def _top_k_accuracy(probabilities: np.ndarray, labels: np.ndarray, *, k: int) -> float:
    probability_matrix = np.asarray(probabilities, dtype=float)
    label_indices = np.asarray(labels, dtype=int).reshape(-1)
    if probability_matrix.size == 0:
        return float("nan")
    if probability_matrix.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional matrix.")
    if label_indices.size != probability_matrix.shape[0]:
        raise ValueError("labels must have one entry per probability row.")
    if probability_matrix.shape[1] == 0:
        return float("nan")

    k_value = min(max(int(k), 1), probability_matrix.shape[1])
    thresholds = np.partition(probability_matrix, -k_value, axis=1)[:, -k_value]
    hits: list[bool] = []
    for row, label, threshold in zip(probability_matrix, label_indices, thresholds, strict=True):
        if label < 0 or label >= row.shape[0]:
            hits.append(False)
        else:
            hits.append(bool(row[int(label)] >= threshold))
    return float(np.mean(hits))


def _numeric_label_indices(values: pd.Series) -> np.ndarray | None:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        return None
    as_float = numeric.to_numpy(dtype=float)
    as_int = as_float.astype(int)
    if not np.allclose(as_float, as_int):
        return None
    return as_int


def _class_name_index_map(group: pd.DataFrame) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for column in group.columns:
        match = _CLASS_COLUMN_RE.fullmatch(str(column))
        if match is None:
            continue
        values = group[column].dropna().astype(str).unique()
        if len(values) == 1:
            mapping[str(values[0])] = int(match.group(1))
    return mapping


def _label_indices_from_group(group: pd.DataFrame) -> np.ndarray:
    if "true_label_index" in group.columns:
        label_indices = _numeric_label_indices(group["true_label_index"])
        if label_indices is not None:
            return label_indices

    if "true_label" not in group.columns:
        raise ValueError("Prediction metrics require true_label or true_label_index.")

    label_indices = _numeric_label_indices(group["true_label"])
    if label_indices is not None:
        return label_indices

    class_index_by_name = _class_name_index_map(group)
    if class_index_by_name:
        resolved: list[int] = []
        for value in group["true_label"].astype(str):
            if value not in class_index_by_name:
                raise ValueError(f"true_label value {value!r} is absent from class_<index> columns.")
            resolved.append(class_index_by_name[value])
        return np.asarray(resolved, dtype=int)

    raise ValueError(
        "Prediction metrics cannot infer numeric class indices from non-numeric true_label values; "
        "include true_label_index or class_<index> columns."
    )


def _probability_class_indices(prob_columns: Sequence[str]) -> np.ndarray:
    return np.asarray([int(str(column).rsplit("_", 1)[-1]) for column in prob_columns], dtype=int)


def _labels_to_probability_positions(label_indices: np.ndarray, prob_class_indices: np.ndarray) -> np.ndarray:
    position_by_class = {int(class_index): position for position, class_index in enumerate(prob_class_indices)}
    positions: list[int] = []
    for label_index in label_indices:
        class_index = int(label_index)
        if class_index not in position_by_class:
            raise ValueError(f"true_label_index {class_index} is absent from probability columns.")
        positions.append(position_by_class[class_index])
    return np.asarray(positions, dtype=int)


def _prediction_metric_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    all_protocols = importlib.import_module("neureptrace.bushmeg_all_protocols")
    prob_columns = all_protocols._probability_columns(predictions)
    if predictions.empty or not prob_columns or ("true_label" not in predictions.columns and "true_label_index" not in predictions.columns):
        return pd.DataFrame()

    prob_class_indices = _probability_class_indices(prob_columns)
    subject_column = "outer_test_subject" if "outer_test_subject" in predictions.columns else "heldout_subject"
    rows: list[dict[str, Any]] = []
    for subject, group in predictions.groupby(subject_column, sort=False):
        probabilities = group[prob_columns].astype(float).to_numpy()
        label_indices = _label_indices_from_group(group)
        labels = _labels_to_probability_positions(label_indices, prob_class_indices)
        predicted = probabilities.argmax(axis=1)
        rows.append(
            {
                "outer_test_subject": str(subject),
                "accuracy": float(accuracy_score(labels, predicted)),
                "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
                "top2_accuracy": _top_k_accuracy(probabilities, labels, k=2),
                "top3_accuracy": _top_k_accuracy(probabilities, labels, k=3),
                "log_loss": float(log_loss(labels, probabilities, labels=np.arange(probabilities.shape[1]))),
                "brier": float(brier_score_multiclass(probabilities, labels)),
                "ece": float(expected_calibration_error(probabilities, labels)),
            }
        )
    return pd.DataFrame(rows)


def install() -> None:
    """Patch all-protocol metric recomputation to respect explicit class indices."""

    all_protocols = importlib.import_module("neureptrace.bushmeg_all_protocols")
    if getattr(all_protocols, _PATCH_MARKER, False):
        return

    all_protocols._top_k_accuracy = _top_k_accuracy
    all_protocols._prediction_metric_frame = _prediction_metric_frame
    setattr(all_protocols, _PATCH_MARKER, True)


__all__ = ["install"]
