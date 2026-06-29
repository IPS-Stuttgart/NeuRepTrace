"""Summarize ensemble and stacked metrics without optional grouping columns."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Sequence
from functools import wraps

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss

from neureptrace.metrics import brier_score_multiclass, expected_calibration_error

_PATCH_MARKER = "_neureptrace_observation_ensemble_ungrouped_metrics_patch_installed"
_STACKING_PATCH_MARKER = "_neureptrace_probability_stacking_ungrouped_metrics_patch_installed"


def _iter_metric_groups(frame: pd.DataFrame, group_columns: Sequence[str]) -> Iterable[tuple[object, pd.DataFrame]]:
    if not group_columns:
        if not frame.empty:
            yield (), frame
        return
    yield from frame.groupby(list(group_columns), dropna=False, sort=True)


def _top_k_accuracy_with_label_values(
    probabilities: np.ndarray,
    true_labels: np.ndarray,
    label_values: Sequence[int],
    *,
    k: int,
    validate_positive_integer: Callable[..., int],
) -> float:
    """Return top-k accuracy with deterministic class-column tie handling."""

    probabilities = np.asarray(probabilities, dtype=float)
    true_labels = np.asarray(true_labels, dtype=int).reshape(-1)
    label_values_array = np.asarray(label_values, dtype=int)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional array.")
    if probabilities.shape[0] != true_labels.shape[0]:
        raise ValueError("probabilities and true_labels must contain the same number of rows.")
    if probabilities.shape[1] != label_values_array.shape[0]:
        raise ValueError("label_values must contain one label per probability column.")
    k = validate_positive_integer(k, name="k")

    effective_k = min(k, probabilities.shape[1])
    if effective_k >= probabilities.shape[1]:
        valid_labels = np.isin(true_labels, label_values_array)
        return float(np.mean(valid_labels))

    top_positions = np.argsort(-probabilities, axis=1, kind="mergesort")[:, :effective_k]
    top_labels = label_values_array[top_positions]
    return float(np.mean(np.any(top_labels == true_labels[:, None], axis=1)))


def _top_k_accuracy_from_label_values(
    probabilities: np.ndarray,
    true_labels: np.ndarray,
    label_values: Sequence[int],
    *,
    k: int,
) -> float:
    observation_ensemble = importlib.import_module("neureptrace.observation_ensemble")
    return _top_k_accuracy_with_label_values(
        probabilities,
        true_labels,
        label_values,
        k=k,
        validate_positive_integer=observation_ensemble._validate_positive_integer,
    )


def _stacking_top_k_accuracy_from_label_values(
    probabilities: np.ndarray,
    true_labels: np.ndarray,
    label_values: Sequence[int],
    *,
    k: int,
) -> float:
    probability_stacking = importlib.import_module("neureptrace.probability_stacking")
    return _top_k_accuracy_with_label_values(
        probabilities,
        true_labels,
        label_values,
        k=k,
        validate_positive_integer=probability_stacking._validate_positive_integer,
    )


def _install_observation_ensemble_patch() -> None:
    observation_ensemble = importlib.import_module("neureptrace.observation_ensemble")
    observation_ensemble._top_k_accuracy_from_label_values = _top_k_accuracy_from_label_values
    original_summarize = observation_ensemble.summarize_ensemble_metrics
    if getattr(original_summarize, _PATCH_MARKER, False):
        return

    @wraps(original_summarize)
    def summarize_ensemble_metrics(observations: pd.DataFrame, *, ece_bins: int = 10) -> pd.DataFrame:
        """Summarize ensemble observation rows, including valid tables without group columns."""

        ece_bins_value = observation_ensemble._validate_positive_integer(ece_bins, name="ece_bins")
        prob_columns = observation_ensemble.probability_columns(observations)
        if "true_label" not in observations.columns or not prob_columns:
            raise ValueError("Ensemble observations must contain true_label and prob_class_* columns.")
        label_values = observation_ensemble._label_values(prob_columns)
        group_columns = [column for column in observation_ensemble._METRIC_GROUP_COLUMNS if column in observations.columns]
        rows: list[dict[str, object]] = []
        for group_key, group in _iter_metric_groups(observations, group_columns):
            if len(group_columns) == 1 and not isinstance(group_key, tuple):
                group_key = (group_key,)
            probabilities = group.loc[:, list(prob_columns)].to_numpy(dtype=float)
            observation_ensemble._validate_probability_matrix(
                probabilities,
                context=f"metric group {dict(zip(group_columns, group_key))}",
                probability_tolerance=observation_ensemble.DEFAULT_PROBABILITY_TOLERANCE,
            )
            true_label_values = observation_ensemble._integer_label_values(group["true_label"])
            true_positions = observation_ensemble._label_positions(true_label_values, label_values)
            prediction_positions = probabilities.argmax(axis=1)
            predicted_label_values = np.asarray([label_values[position] for position in prediction_positions], dtype=int)
            row = dict(zip(group_columns, group_key))
            row.update(
                {
                    "accuracy": accuracy_score(true_label_values, predicted_label_values),
                    "balanced_accuracy": balanced_accuracy_score(true_label_values, predicted_label_values),
                    "top2_accuracy": observation_ensemble._top_k_accuracy_from_label_values(probabilities, true_label_values, label_values, k=2),
                    "top3_accuracy": observation_ensemble._top_k_accuracy_from_label_values(probabilities, true_label_values, label_values, k=3),
                    "log_loss": log_loss(true_label_values, probabilities, labels=list(label_values)),
                    "brier": brier_score_multiclass(probabilities, true_positions),
                    "ece": expected_calibration_error(probabilities, true_positions, n_bins=ece_bins_value),
                    "n_train": "",
                    "n_test": int(len(group)),
                    "n_classes": int(len(prob_columns)),
                    "class_names": "|".join(str(group.iloc[0].get(f"class_{label}", label)) for label in label_values),
                }
            )
            for column in observation_ensemble._METRIC_PROVENANCE_COLUMNS:
                if column not in group.columns:
                    continue
                values = group[column].drop_duplicates()
                if len(values) == 1:
                    row[column] = values.iloc[0]
            rows.append(row)
        return pd.DataFrame(rows)

    setattr(summarize_ensemble_metrics, _PATCH_MARKER, True)
    observation_ensemble.summarize_ensemble_metrics = summarize_ensemble_metrics


def _install_probability_stacking_patch() -> None:
    probability_stacking = importlib.import_module("neureptrace.probability_stacking")
    probability_stacking._top_k_accuracy_from_label_values = _stacking_top_k_accuracy_from_label_values
    original_summarize = probability_stacking.summarize_stacked_metrics
    if getattr(original_summarize, _STACKING_PATCH_MARKER, False):
        return

    @wraps(original_summarize)
    def summarize_stacked_metrics(observations: pd.DataFrame) -> pd.DataFrame:
        """Summarize stacked observation rows, including valid tables without group columns."""

        prob_columns = probability_stacking.probability_columns(observations)
        if "true_label" not in observations.columns or not prob_columns:
            raise ValueError("Stacked observations must contain true_label and prob_class_* columns.")
        label_values = probability_stacking._label_values(prob_columns)
        label_value_set = set(label_values)
        group_columns = [column for column in probability_stacking._METRIC_GROUP_COLUMNS if column in observations.columns]
        rows: list[dict[str, object]] = []
        for group_key, group in _iter_metric_groups(observations, group_columns):
            if len(group_columns) == 1 and not isinstance(group_key, tuple):
                group_key = (group_key,)
            group_context = dict(zip(group_columns, group_key, strict=True))
            probabilities = probability_stacking._validate_probability_matrix(
                group.loc[:, list(prob_columns)].to_numpy(dtype=float),
                context=f"Probability values for metric group {group_context}",
            )
            true_label_values = probability_stacking._integer_label_array(group["true_label"], name="true_label")
            missing_labels = sorted(set(int(label) for label in true_label_values if int(label) not in label_value_set))
            if missing_labels:
                raise ValueError(f"true_label values must index probability labels {list(label_values)}; missing labels: {missing_labels[:5]}")
            true_positions = probability_stacking._label_positions(true_label_values, label_values)
            predicted_label_values = np.asarray([label_values[position] for position in probabilities.argmax(axis=1)], dtype=int)
            row = dict(zip(group_columns, group_key, strict=True))
            row.update(
                {
                    "accuracy": float(accuracy_score(true_label_values, predicted_label_values)),
                    "balanced_accuracy": float(balanced_accuracy_score(true_label_values, predicted_label_values)),
                    "top2_accuracy": probability_stacking._top_k_accuracy_from_label_values(probabilities, true_label_values, label_values, k=2),
                    "top3_accuracy": probability_stacking._top_k_accuracy_from_label_values(probabilities, true_label_values, label_values, k=3),
                    "log_loss": float(log_loss(true_label_values, probabilities, labels=list(label_values))),
                    "brier": float(brier_score_multiclass(probabilities, true_positions)),
                    "ece": float(expected_calibration_error(probabilities, true_positions)),
                    "n_test": int(len(group)),
                    "n_classes": int(len(prob_columns)),
                }
            )
            for column in probability_stacking._STACKING_METADATA_COLUMNS:
                if column not in group.columns:
                    row[column] = ""
                    continue
                values = group[column].drop_duplicates()
                if len(values) > 1:
                    raise ValueError(f"Stacking metadata column {column!r} is inconsistent within metric group {group_context}.")
                row[column] = "" if values.empty else values.iloc[0]
            rows.append(row)
        return pd.DataFrame(rows)

    setattr(summarize_stacked_metrics, _STACKING_PATCH_MARKER, True)
    probability_stacking.summarize_stacked_metrics = summarize_stacked_metrics


def install() -> None:
    """Patch metric summaries so ungrouped tables produce one global row."""

    importlib.import_module("neureptrace._observation_ensemble_string_groups_patch").install()
    _install_observation_ensemble_patch()
    _install_probability_stacking_patch()


__all__ = ["install"]
