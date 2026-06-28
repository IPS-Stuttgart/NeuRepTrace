"""Allow BUSH-MEG all-protocol metric recomputation for named labels."""

from __future__ import annotations

import importlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss

from neureptrace.metrics import brier_score_multiclass, expected_calibration_error

_PATCH_MARKER = "_neureptrace_bushmeg_all_protocols_prediction_metric_patch_installed"
_CLASS_COLUMN_RE = re.compile(r"^class_(\d+)$")
_METRIC_JOIN_COLUMNS = (
    "fold_index",
    "target_calibration_per_class",
    "k_per_class",
    "target_calibration_seed",
)
_METRIC_COLUMNS = ("accuracy", "balanced_accuracy", "top2_accuracy", "top3_accuracy", "log_loss", "brier", "ece")


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
    if not np.isfinite(as_float).all():
        return None
    rounded = np.rint(as_float)
    if not np.isclose(as_float, rounded, rtol=0.0, atol=1.0e-12).all():
        return None
    return rounded.astype(int)


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


def _label_indices_from_named_column(group: pd.DataFrame, column: str) -> np.ndarray | None:
    if column not in group.columns:
        return None

    label_indices = _numeric_label_indices(group[column])
    if label_indices is not None:
        return label_indices

    class_index_by_name = _class_name_index_map(group)
    if class_index_by_name:
        resolved: list[int] = []
        for value in group[column].astype(str):
            if value not in class_index_by_name:
                raise ValueError(f"{column} value {value!r} is absent from class_<index> columns.")
            resolved.append(class_index_by_name[value])
        return np.asarray(resolved, dtype=int)
    return None


def _label_indices_from_group(group: pd.DataFrame) -> np.ndarray:
    if "true_label_index" in group.columns:
        label_indices = _numeric_label_indices(group["true_label_index"])
        if label_indices is not None:
            return label_indices

    if "true_label" not in group.columns:
        raise ValueError("Prediction metrics require true_label or true_label_index.")

    label_indices = _label_indices_from_named_column(group, "true_label")
    if label_indices is not None:
        return label_indices

    raise ValueError(
        "Prediction metrics cannot infer numeric class indices from non-numeric true_label values; "
        "include true_label_index or class_<index> columns."
    )


def _raw_label_values(group: pd.DataFrame, *, index_column: str, label_column: str, role: str) -> np.ndarray:
    if index_column in group.columns:
        numeric = _numeric_label_indices(group[index_column])
        if numeric is not None:
            return numeric
    if label_column in group.columns:
        return group[label_column].astype(object).to_numpy()
    raise ValueError(f"Prediction metrics require {label_column} or {index_column} for {role} labels.")


def _label_only_metric_vectors(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return comparable true/predicted vectors when probabilities are absent."""

    true_indices = None
    predicted_indices = None
    if "true_label_index" in group.columns or "true_label" in group.columns:
        true_indices = _label_indices_from_named_column(group, "true_label")
        if true_indices is None and "true_label_index" in group.columns:
            true_indices = _numeric_label_indices(group["true_label_index"])
    if "predicted_label_index" in group.columns or "predicted_label" in group.columns:
        predicted_indices = _label_indices_from_named_column(group, "predicted_label")
        if predicted_indices is None and "predicted_label_index" in group.columns:
            predicted_indices = _numeric_label_indices(group["predicted_label_index"])

    if true_indices is not None and predicted_indices is not None:
        return true_indices, predicted_indices

    true_values = _raw_label_values(group, index_column="true_label_index", label_column="true_label", role="true")
    predicted_values = _raw_label_values(
        group,
        index_column="predicted_label_index",
        label_column="predicted_label",
        role="predicted",
    )
    if true_values.shape[0] != predicted_values.shape[0]:
        raise ValueError("Prediction metrics require true and predicted labels to have the same row count.")
    return true_values.astype(str), predicted_values.astype(str)


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


def _subject_column(frame: pd.DataFrame) -> str:
    if "outer_test_subject" in frame.columns:
        return "outer_test_subject"
    if "heldout_subject" in frame.columns:
        return "heldout_subject"
    raise ValueError("Prediction metrics require outer_test_subject or heldout_subject.")


def _prediction_group_columns(predictions: pd.DataFrame) -> list[str]:
    columns = [_subject_column(predictions)]
    columns.extend(column for column in _METRIC_JOIN_COLUMNS if column in predictions.columns)
    return columns


def _prediction_metric_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    all_protocols = importlib.import_module("neureptrace.bushmeg_all_protocols")
    prob_columns = all_protocols._probability_columns(predictions)
    has_truth = "true_label" in predictions.columns or "true_label_index" in predictions.columns
    has_predictions = "predicted_label" in predictions.columns or "predicted_label_index" in predictions.columns
    if predictions.empty or not has_truth or (not prob_columns and not has_predictions):
        return pd.DataFrame()

    prob_class_indices = _probability_class_indices(prob_columns) if prob_columns else np.asarray([], dtype=int)
    group_columns = _prediction_group_columns(predictions)
    rows: list[dict[str, Any]] = []
    groupby_key: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for group_key, group in predictions.groupby(groupby_key, sort=False, dropna=False):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        row = {column: value for column, value in zip(group_columns, key_values, strict=True)}
        row["outer_test_subject"] = str(row.pop("heldout_subject", row.get("outer_test_subject", "")))

        if prob_columns:
            probabilities = group[prob_columns].astype(float).to_numpy()
            label_indices = _label_indices_from_group(group)
            labels = _labels_to_probability_positions(label_indices, prob_class_indices)
            predicted = probabilities.argmax(axis=1)
            rows.append(
                {
                    **row,
                    "accuracy": float(accuracy_score(labels, predicted)),
                    "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
                    "top2_accuracy": _top_k_accuracy(probabilities, labels, k=2),
                    "top3_accuracy": _top_k_accuracy(probabilities, labels, k=3),
                    "log_loss": float(log_loss(labels, probabilities, labels=np.arange(probabilities.shape[1]))),
                    "brier": float(brier_score_multiclass(probabilities, labels)),
                    "ece": float(expected_calibration_error(probabilities, labels)),
                }
            )
        else:
            labels, predicted = _label_only_metric_vectors(group)
            rows.append(
                {
                    **row,
                    "accuracy": float(accuracy_score(labels, predicted)),
                    "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
                    "top2_accuracy": np.nan,
                    "top3_accuracy": np.nan,
                    "log_loss": np.nan,
                    "brier": np.nan,
                    "ece": np.nan,
                }
            )
    return pd.DataFrame(rows)


def _summary_prediction_key_pairs(summary: pd.DataFrame, prediction_metrics: pd.DataFrame) -> list[tuple[str, str]]:
    summary_subject = "outer_test_subject" if "outer_test_subject" in summary.columns else "heldout_subject"
    metric_subject = "outer_test_subject" if "outer_test_subject" in prediction_metrics.columns else "heldout_subject"
    pairs = [(summary_subject, metric_subject)]
    pairs.extend((column, column) for column in _METRIC_JOIN_COLUMNS if column in summary.columns and column in prediction_metrics.columns)
    return pairs


def _merge_prediction_metrics(summary: pd.DataFrame, prediction_metrics: pd.DataFrame) -> pd.DataFrame:
    if prediction_metrics.empty:
        return summary
    joined = summary.copy()
    metrics = prediction_metrics.copy()
    key_pairs = _summary_prediction_key_pairs(joined, metrics)
    join_keys: list[str] = []
    for index, (left_column, right_column) in enumerate(key_pairs):
        key = f"__prediction_metric_join_{index}"
        joined[key] = joined[left_column].astype(str)
        metrics[key] = metrics[right_column].astype(str)
        join_keys.append(key)
    merged = joined.merge(metrics, how="left", on=join_keys, suffixes=("", "_from_predictions"))
    merged = merged.drop(columns=join_keys, errors="ignore")
    for left_column, right_column in key_pairs:
        fallback = f"{right_column}_from_predictions"
        if fallback in merged.columns:
            merged = merged.drop(columns=[fallback])
        if right_column != left_column and right_column in merged.columns and right_column not in summary.columns:
            merged = merged.drop(columns=[right_column])
    return merged


def _normalize_summary(
    raw_summary: pd.DataFrame,
    raw_predictions: pd.DataFrame,
    *,
    spec: Any,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    all_protocols = importlib.import_module("neureptrace.bushmeg_all_protocols")
    if raw_summary.empty:
        return pd.DataFrame(columns=all_protocols.SUMMARY_COLUMNS)
    summary = raw_summary.copy()
    if "analysis" in summary.columns and (summary["analysis"] == "temporal_ensemble").any():
        summary = summary.loc[summary["analysis"] == "temporal_ensemble"].copy()
    prediction_metrics = _prediction_metric_frame(raw_predictions)
    if not prediction_metrics.empty:
        summary = _merge_prediction_metrics(summary, prediction_metrics)
        for metric in _METRIC_COLUMNS:
            fallback = f"{metric}_from_predictions"
            if fallback in summary.columns:
                if metric not in summary.columns:
                    summary[metric] = summary[fallback]
                else:
                    summary[metric] = summary[metric].where(pd.notna(summary[metric]), summary[fallback])
                summary = summary.drop(columns=[fallback])
    metadata = spec.protocol.metadata()
    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        n_target_trials = all_protocols._first_existing(row, ("n_test_trials", "n_test", "n_target_trials"))
        protocol_category = int(spec.protocol_category)
        n_calibration_trials = (
            n_target_trials
            if protocol_category == 4 and pd.notna(n_target_trials)
            else all_protocols._first_existing(row, ("n_calibration_trials", "n_target_calibration_trials"), 0)
        )
        target_calibration_per_class = all_protocols._first_existing(row, ("target_calibration_per_class", "few_shot_target_calibration_per_class"))
        k_per_class = all_protocols._first_existing(row, ("k_per_class", "target_calibration_per_class", "few_shot_target_calibration_per_class"))
        n_target_calibration_trials = all_protocols._first_existing(row, ("n_target_calibration_trials", "n_calibration_trials"), n_calibration_trials)
        n_target_evaluation_trials = all_protocols._first_existing(row, ("n_target_evaluation_trials", "n_evaluation_trials"))
        target_calibration_seed = all_protocols._first_existing(row, ("target_calibration_seed", "few_shot_target_calibration_seed"))
        normalized = {
            "method": spec.method,
            "method_family": spec.method_family,
            **metadata,
            "calibration_rows_disjoint_from_evaluation": all_protocols._first_existing(
                row,
                ("calibration_rows_disjoint_from_evaluation",),
                metadata["calibration_rows_disjoint_from_evaluation"],
            ),
            "outer_test_subject": str(all_protocols._first_existing(row, ("outer_test_subject", "heldout_subject"))),
            "n_source_subjects": all_protocols._first_existing(row, ("n_train_subjects", "n_source_subjects")),
            "n_source_trials": all_protocols._first_existing(row, ("n_train", "n_source_trials")),
            "n_target_trials": n_target_trials,
            "n_calibration_trials": n_calibration_trials,
            "target_calibration_per_class": target_calibration_per_class,
            "k_per_class": k_per_class,
            "n_target_calibration_trials": n_target_calibration_trials,
            "n_target_evaluation_trials": n_target_evaluation_trials,
            "target_calibration_seed": target_calibration_seed,
            "feature_kind": all_protocols._first_existing(
                row,
                ("feature_kind", "covariance_feature_mode", "window_feature_mode", "feature_preprocessor"),
                spec.method_family,
            ),
            "window_centers": all_protocols._first_existing(row, ("window_centers", "time")),
            "window_size": all_protocols._window_size_from_row(row, config),
            "temporal_bins": all_protocols._first_existing(row, ("temporal_bins",), pd.NA),
            "balanced_accuracy": all_protocols._first_existing(row, ("balanced_accuracy",)),
            "accuracy": all_protocols._first_existing(row, ("accuracy",)),
            "top2_accuracy": all_protocols._first_existing(row, ("top2_accuracy",)),
            "top3_accuracy": all_protocols._first_existing(row, ("top3_accuracy",)),
            "log_loss": all_protocols._first_existing(row, ("log_loss",)),
            "brier": all_protocols._first_existing(row, ("brier",)),
            "ece": all_protocols._first_existing(row, ("ece",)),
        }
        for column, value in row.items():
            normalized.setdefault(str(column), value)
        rows.append(normalized)
    frame = pd.DataFrame(rows)
    extra_columns = [column for column in frame.columns if column not in all_protocols.SUMMARY_COLUMNS]
    return frame[all_protocols.SUMMARY_COLUMNS + extra_columns]


def install() -> None:
    """Patch all-protocol metric recomputation to respect explicit class indices."""

    all_protocols = importlib.import_module("neureptrace.bushmeg_all_protocols")
    if getattr(all_protocols, _PATCH_MARKER, False):
        return

    all_protocols._top_k_accuracy = _top_k_accuracy
    all_protocols._prediction_metric_frame = _prediction_metric_frame
    all_protocols._normalize_summary = _normalize_summary
    setattr(all_protocols, _PATCH_MARKER, True)


__all__ = ["install"]
