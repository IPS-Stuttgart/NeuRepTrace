"""Allow probability-stacking metric summaries without grouping columns."""

from __future__ import annotations

import numpy as np
import pandas as pd

_INSTALLED = False
_ORIGINAL_SUMMARIZE_STACKED_METRICS = None


def _summarize_global_metrics(ps, observations: pd.DataFrame) -> pd.DataFrame:
    """Return one global metrics row when no metric grouping columns exist."""

    prob_columns = ps.probability_columns(observations)
    if "true_label" not in observations.columns or not prob_columns:
        raise ValueError("Stacked observations must contain true_label and prob_class_* columns.")

    label_values = ps._label_values(prob_columns)
    label_value_set = set(label_values)
    probabilities = ps._validate_probability_matrix(
        observations.loc[:, list(prob_columns)].to_numpy(dtype=float),
        context="Probability values for global metric summary",
    )
    true_label_values = ps._integer_label_array(observations["true_label"], name="true_label")
    missing_labels = sorted(set(int(label) for label in true_label_values if int(label) not in label_value_set))
    if missing_labels:
        raise ValueError(f"true_label values must index probability labels {list(label_values)}; missing labels: {missing_labels[:5]}")

    true_positions = ps._label_positions(true_label_values, label_values)
    predicted_label_values = np.asarray([label_values[position] for position in probabilities.argmax(axis=1)], dtype=int)
    row: dict[str, object] = {
        "accuracy": float(ps.accuracy_score(true_label_values, predicted_label_values)),
        "balanced_accuracy": float(ps.balanced_accuracy_score(true_label_values, predicted_label_values)),
        "top2_accuracy": ps._top_k_accuracy_from_label_values(probabilities, true_label_values, label_values, k=2),
        "top3_accuracy": ps._top_k_accuracy_from_label_values(probabilities, true_label_values, label_values, k=3),
        "log_loss": float(ps.log_loss(true_label_values, probabilities, labels=list(label_values))),
        "brier": float(ps.brier_score_multiclass(probabilities, true_positions)),
        "ece": float(ps.expected_calibration_error(probabilities, true_positions)),
        "n_test": int(len(observations)),
        "n_classes": int(len(prob_columns)),
    }
    for column in ps._STACKING_METADATA_COLUMNS:
        if column not in observations.columns:
            row[column] = ""
            continue
        values = observations[column].drop_duplicates()
        if len(values) > 1:
            raise ValueError(f"Stacking metadata column {column!r} is inconsistent within the global metric summary.")
        row[column] = "" if values.empty else values.iloc[0]
    return pd.DataFrame([row])


def install() -> None:
    """Install the ungrouped-summary wrapper once."""

    global _INSTALLED, _ORIGINAL_SUMMARIZE_STACKED_METRICS
    if _INSTALLED:
        return

    from neureptrace import probability_stacking as ps

    _ORIGINAL_SUMMARIZE_STACKED_METRICS = ps.summarize_stacked_metrics

    def _summarize_stacked_metrics(observations: pd.DataFrame) -> pd.DataFrame:
        group_columns = [column for column in ps._METRIC_GROUP_COLUMNS if column in observations.columns]
        if group_columns:
            return _ORIGINAL_SUMMARIZE_STACKED_METRICS(observations)
        return _summarize_global_metrics(ps, observations)

    _summarize_stacked_metrics.__name__ = _ORIGINAL_SUMMARIZE_STACKED_METRICS.__name__
    _summarize_stacked_metrics.__doc__ = _ORIGINAL_SUMMARIZE_STACKED_METRICS.__doc__
    ps.summarize_stacked_metrics = _summarize_stacked_metrics
    _INSTALLED = True
