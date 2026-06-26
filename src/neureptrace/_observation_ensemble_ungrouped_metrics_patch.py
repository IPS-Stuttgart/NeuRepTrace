"""Summarize ensemble metrics without optional grouping columns."""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Sequence
from functools import wraps

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss

from neureptrace.metrics import brier_score_multiclass, expected_calibration_error

_PATCH_MARKER = "_neureptrace_observation_ensemble_ungrouped_metrics_patch_installed"


def _iter_metric_groups(frame: pd.DataFrame, group_columns: Sequence[str]) -> Iterable[tuple[object, pd.DataFrame]]:
    if not group_columns:
        if not frame.empty:
            yield (), frame
        return
    yield from frame.groupby(list(group_columns), dropna=False, sort=True)


def install() -> None:
    """Patch ensemble metric summaries so ungrouped tables produce one global row."""

    importlib.import_module("neureptrace._observation_ensemble_string_groups_patch").install()

    observation_ensemble = importlib.import_module("neureptrace.observation_ensemble")
    original_summarize = observation_ensemble.summarize_ensemble_metrics
    if getattr(original_summarize, _PATCH_MARKER, False):
        return

    @wraps(original_summarize)
    def summarize_ensemble_metrics(observations: pd.DataFrame, *, ece_bins: int = 10) -> pd.DataFrame:
        """Summarize ensemble observation rows, including valid tables without group columns."""

        if ece_bins < 1:
            raise ValueError("ece_bins must be positive.")
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
                    "ece": expected_calibration_error(probabilities, true_positions, n_bins=ece_bins),
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


__all__ = ["install"]
